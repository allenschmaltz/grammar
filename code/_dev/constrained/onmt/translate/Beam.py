from __future__ import division
import torch
from collections import namedtuple

Hypothesis = namedtuple("Hypothesis", ["allowed_action", "correction_diff", "uncovered_src_index"])


class Beam(object):
    """
    Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """
    def __init__(self, size, pad, bos, eos, unk,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # For constrained decoding:
        self.src_raw = []
        self.src_as_tgt = []  # target vocab ids of source

        self._pad = pad
        self._bos = bos
        self._unk = unk

        self._tag_weight = None

        self._del_start_sym_id = None
        self._del_end_sym_id = None
        self._ins_start_sym_id = None
        self._ins_end_sym_id = None

        self._del_start_sym = None
        self._del_end_sym = None
        self._ins_start_sym = None
        self._ins_end_sym = None


        self._action_next = -2 # <ins>, <del>, copy
        self._action_in_del = -3  # copy or </del>
        self._action_in_ins = -4  # anything except <ins>, </del>, PAD, BOS, EOS, UNK
        self._action_complete = -5  # <ins>, EOS
        self._action_terminate = -6 # EOS has been seen or an exhausted hyps

        self.states = []  # every timestep is a list (of length <= beam_size) of Hypothesis objects


    def set_beam_constants(self, tag_weight, del_start_sym_id, del_end_sym_id, ins_start_sym_id, ins_end_sym_id, del_start_sym, del_end_sym, ins_start_sym, ins_end_sym):
        self._tag_weight = tag_weight
        self._del_start_sym_id = del_start_sym_id
        self._del_end_sym_id = del_end_sym_id
        self._ins_start_sym_id = ins_start_sym_id
        self._ins_end_sym_id = ins_end_sym_id
        self._del_start_sym = del_start_sym
        self._del_end_sym = del_end_sym
        self._ins_start_sym = ins_start_sym
        self._ins_end_sym = ins_end_sym

    def initialize_states(self, beam_size, tgt_vocab):
        new_hyp_states = []
        for k in range(0, beam_size):
            if k == 0:
                new_hyp_states.append(Hypothesis(self._action_next, tgt_vocab.itos[self._bos], 0))
            else: # null states:
                new_hyp_states.append(Hypothesis(self._action_terminate, "", len(self.src_as_tgt)))
        self.states.append(new_hyp_states)


    def _get_one_allowed_transition_mask(self, num_words, state):

        allowed_action = state.allowed_action
        src_index = state.uncovered_src_index
        if allowed_action == self._action_next:
            allowed_action_mask = self.tt.ByteTensor(num_words).fill_(1)
            allowed_action_mask[self._del_start_sym_id] = 0
            allowed_action_mask[self._ins_start_sym_id] = 0
            assert src_index <= len(self.src_as_tgt) - 1
            allowed_action_mask[self.src_as_tgt[src_index]] = 0
        elif allowed_action == self._action_complete:
            allowed_action_mask = self.tt.ByteTensor(num_words).fill_(1)
            assert src_index == len(self.src_as_tgt)
            allowed_action_mask[self._eos] = 0
            allowed_action_mask[self._ins_start_sym_id] = 0  # insertion at the end of a sentence
        elif allowed_action == self._action_in_del:
            allowed_action_mask = self.tt.ByteTensor(num_words).fill_(1)
            if src_index <= len(self.src_as_tgt) - 1:  # otherwise, source has been exhausted, so the only option is to end del
                allowed_action_mask[self.src_as_tgt[src_index]] = 0
            allowed_action_mask[self._del_end_sym_id] = 0
        elif allowed_action == self._action_in_ins:  # any word and </ins>; insertion of <unk> not allowed
            allowed_action_mask = self.tt.ByteTensor(num_words).fill_(0)
            allowed_action_mask[self._del_start_sym_id] = 1
            allowed_action_mask[self._del_end_sym_id] = 1
            allowed_action_mask[self._ins_start_sym_id] = 1
            allowed_action_mask[self._bos] = 1
            allowed_action_mask[self._eos] = 1
            allowed_action_mask[self._pad] = 1
            allowed_action_mask[self._unk] = 1  # TODO: when intersecting with LMs, allow unk insertion
        elif allowed_action == self._action_terminate:  # everything is masked
            allowed_action_mask = self.tt.ByteTensor(num_words).fill_(1)

        return allowed_action_mask

    def _get_allowed_transition_masks(self, word_probs):
        """
        Mask output softmax indecies based on current state

        :param word_probs:
        :return:
        """
        assert len(self.next_ys) == len(self.states)

        for k in range(len(word_probs)):
            allowed_action_mask = self._get_one_allowed_transition_mask(word_probs.size(1), self.states[-1][k])
            word_probs[k].masked_fill_(allowed_action_mask, -1e20)

        return word_probs


    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out, tgt_vocab):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)

        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20


        # update the tag weights:
        for k in range(len(word_probs)):
            word_probs[k][self._ins_start_sym_id] += self._tag_weight
            word_probs[k][self._ins_end_sym_id] += self._tag_weight
            word_probs[k][self._del_start_sym_id] += self._tag_weight
            word_probs[k][self._del_end_sym_id] += self._tag_weight


        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                self.scores.unsqueeze(1).expand_as(word_probs)

            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20

            beam_scores = self._get_allowed_transition_masks(beam_scores)
        else:
            word_probs = self._get_allowed_transition_masks(word_probs)
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)

        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                            True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words

        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))
        self.attn.append(attn_out.index_select(0, prev_k))

        # zero-out hyps for which the beam was exhausted:
        proposed_next_ys = self.next_ys[-1]
        for i in range(0, len(proposed_next_ys)):
            if best_scores[i] < -1e19:
                proposed_next_ys[i] = self._pad
        self.next_ys[-1] = proposed_next_ys



        if self.global_scorer is not None:
            self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos and self.states[-1][self.prev_ks[-1][i]].allowed_action == self._action_complete:
                s = self.scores[i]
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score, provided top-of-beam is in the self._action_complete state.
        if self.next_ys[-1][0] == self._eos and self.states[-1][self.prev_ks[-1][0]].allowed_action == self._action_complete:
            # self.all_scores.append(self.scores)
            self.eos_top = True


        # update states, based on new predictions and previous allowed actions
        new_hyp_states = []
        for i in range(self.next_ys[-1].size(0)):
            prev_k = self.prev_ks[-1][i]
            prev_state = self.states[-1][prev_k]
            pred_y = self.next_ys[-1][i]

            if best_scores[i] < -1e19:  # in these cases, do not rely on prev_k
                new_hyp_states.append(Hypothesis(self._action_terminate, "", len(self.src_as_tgt)))
            else:
                if prev_state.allowed_action == self._action_complete:
                    if pred_y == self._eos:
                        new_hyp_states.append(Hypothesis(self._action_terminate, "", len(self.src_as_tgt)))
                    elif pred_y == self._ins_start_sym_id:  # start ins
                        new_hyp_states.append(Hypothesis(self._action_in_ins, tgt_vocab.itos[pred_y], prev_state.uncovered_src_index))
                elif prev_state.allowed_action == self._action_next:
                    if pred_y == self._del_start_sym_id:  # start del
                        new_hyp_states.append(Hypothesis(self._action_in_del, tgt_vocab.itos[pred_y], prev_state.uncovered_src_index))
                    elif pred_y == self._ins_start_sym_id:  # start ins
                        new_hyp_states.append(Hypothesis(self._action_in_ins, tgt_vocab.itos[pred_y], prev_state.uncovered_src_index))
                    else:  # copy
                        pred_y_string = self.src_raw[prev_state.uncovered_src_index]
                        next_uncovered_src_index = prev_state.uncovered_src_index + 1
                        if next_uncovered_src_index <= len(self.src_raw) - 1:
                            new_hyp_states.append(Hypothesis(self._action_next, pred_y_string, next_uncovered_src_index))
                        else:  # source is exhausted, so transition to complete state
                            new_hyp_states.append(Hypothesis(self._action_complete, pred_y_string, next_uncovered_src_index))
                elif prev_state.allowed_action == self._action_terminate:
                    assert False, "Invalid state reached"
                elif prev_state.allowed_action == self._action_in_del:
                    if pred_y == self._del_end_sym_id:  # end del
                        if prev_state.uncovered_src_index <= len(self.src_raw) - 1:
                            new_hyp_states.append(Hypothesis(self._action_next, tgt_vocab.itos[pred_y], prev_state.uncovered_src_index))
                        else:
                            new_hyp_states.append(
                                Hypothesis(self._action_complete, tgt_vocab.itos[pred_y], prev_state.uncovered_src_index))
                    else:
                        new_hyp_states.append(Hypothesis(self._action_in_del, self.src_raw[prev_state.uncovered_src_index], prev_state.uncovered_src_index + 1))
                elif prev_state.allowed_action == self._action_in_ins:
                    if pred_y == self._ins_end_sym_id:
                        if prev_state.uncovered_src_index <= len(self.src_raw) - 1:
                            new_hyp_states.append(Hypothesis(self._action_next, tgt_vocab.itos[pred_y], prev_state.uncovered_src_index))
                        else:
                            new_hyp_states.append(Hypothesis(self._action_complete, tgt_vocab.itos[pred_y], prev_state.uncovered_src_index))
                    else:
                        new_hyp_states.append(Hypothesis(self._action_in_ins, tgt_vocab.itos[pred_y], prev_state.uncovered_src_index))

        assert len(new_hyp_states) == self.next_ys[-1].size(0)
        self.states.append(new_hyp_states)


    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best


    def sort_finished(self, minimum=None):
        # NOTE: for constrained decoding, only hyps ending in eos are considered (so self.n_best hypotheses may not always be found):

        # if minimum is not None:
        #     i = 0
        #     # Add from beam until we have minimum outputs.
        #     while len(self.finished) < minimum:
        #         s = self.scores[i]
        #         if self.global_scorer is not None:
        #             global_scores = self.global_scorer.score(self, self.scores)
        #             s = global_scores[i]
        #         self.finished.append((s, len(self.next_ys) - 1, i))
        #         i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        constrained_hyp = []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j+1][k])
            constrained_hyp.append(self.states[j+1][k].correction_diff)
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1]), constrained_hyp[::-1]


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def score(self, beam, logprobs):
        "Additional term add to log probability"
        cov = beam.global_state["coverage"]
        pen = self.beta * torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        l_term = (((5 + len(beam.next_ys)) ** self.alpha) /
                  ((5 + 1) ** self.alpha))
        return (logprobs / l_term) + pen

    def update_global_state(self, beam):
        "Keeps the coverage vector as sum of attens"
        if len(beam.prev_ks) == 1:
            beam.global_state["coverage"] = beam.attn[-1]
        else:
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])
