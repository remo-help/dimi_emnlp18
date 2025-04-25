import torch
import numpy as np
import pickle as pkl
import argparse


def start_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Imports a saved model and translates it into a readable PCFG')
    parser.add_argument("--path",
                        help="The path to the model you wish to translate.",
                        required=True)
    parser.add_argument("--reduce",
                        help="This flag will return a PCFG that has its low-probability transitions pruned for"
                             "readability. This interacts with the --rounding_acc parameter",
                        required=False, default=False, action='store_true')
    parser.add_argument("--outputfile",
                        help="The name of your outputfile. Default: pcfg.pkl",
                        required=False, default='pcfg.pkl')
    parser.add_argument("--rounding_acc",
                        help="Regulates the accuracy to which probabilities are rounded under --reduce"
                             "only enforced if --reduce is called",
                        required=False, default=3, type=int)
    return parser

def load_model(path):
    pcfg = torch.load(path, weights_only=False)
    word_dict = pcfg[0].word_dict
    new_pcfg = pcfg[0].get_current_pcfg()
    return new_pcfg, word_dict


def reduce_transitions(non_terminal_dict, rounding_acc=3, select_acc=0.001):
    """
    we need this function because the saved PCFG has transitions from all non terminals to all other
    non terminals. Here we reduce this to only those transitions that are probable
    :param select_acc:
    :param rounding_acc:
    :param non_terminal_dict:
    :return:
    """
    reduced_non_terms = dict()
    for non_terminal_node in non_terminal_dict:
        reduced_non_terminal ={}
        non_terminal_node_dict = non_terminal_dict[non_terminal_node]
        lookup_table = {idx: non_terminal for idx, non_terminal in enumerate(non_terminal_node_dict.keys())}
        probs_arr = np.array([non_terminal_node_dict[non_terminal] for non_terminal in non_terminal_node_dict.keys()])
        # we round the probs so we dont end up with a huge number of potential transitions
        rounded_probs = np.round(probs_arr, rounding_acc)
        # we select only those probs that survive rounding
        index_arr = np.where(rounded_probs > select_acc)
        for index in index_arr[0]:
            #we grab the rounded probability and the actual label of the transition from our
            #lookup table and create a new transition dictionary
            reduced_non_terminal[lookup_table[index]] = probs_arr[index]
        reduced_non_terms[non_terminal_node]=reduced_non_terminal
    return reduced_non_terms


def create_start_state(reduced_non_terms, start_dist, rounding_acc=3, select_acc=0.001):
    rounded_start_dist = np.round(start_dist, rounding_acc)
    index_arr = np.where(rounded_start_dist > select_acc)[0]
    transitions = {}
    for state in index_arr:
        prob_weight = rounded_start_dist[state]
        for transition in reduced_non_terms[state].keys():
            transition_prob = np.multiply(prob_weight, reduced_non_terms[state][transition])
            if transition in transitions.keys():
                transitions[transition] += transition_prob
            else:
                transitions[transition] = transition_prob
    return transitions

def create_start_state_no_reduction(non_terms, start_dist):
    # creates a start state without reducing possible transitions, this is used when we expect
    # unseen events
    transitions = {}
    # we first need to select only the actual probs from the start dist as the actual
    # array is larger then the number of states
    index_arr = np.flatnonzero(start_dist)
    start_dist_real = start_dist.ravel()[index_arr]
    for state in range(len(start_dist_real)):
        prob_weight = start_dist_real[state]
        for transition in non_terms[state].keys():
            transition_prob = np.multiply(prob_weight, non_terms[state][transition])
            if transition in transitions.keys():
                transitions[transition] += transition_prob
            else:
                transitions[transition] = transition_prob
    return transitions


def translate_to_pcfg(reduced_non_terms, start_state, word_dict, start_state_label='S0'):
    pcfg = {}
    for non_term in reduced_non_terms:
        transition_list = []
        for transition in reduced_non_terms[non_term]:
            #print(transition)
            #print(reduced_non_terms[non_term][transition])
            if type(transition) is int:
                transition_list.append(((word_dict[transition],),reduced_non_terms[non_term][transition]))
            else:
                transition_list.append((transition, reduced_non_terms[non_term][transition]))
        pcfg[non_term] = transition_list
    start_transitions = []
    for transition in start_state:
        if type(transition) is int:
            start_transitions.append(((word_dict[transition],), start_state[transition]))
        else:
            start_transitions.append((transition, start_state[transition]))
    pcfg[start_state_label] = start_transitions
    terms = {}
    for value in word_dict.values():
        terms[value] = {'features': [None], 'rules': None}
    return pcfg, terms


def pcfg_from_path(path, reduce=False, rounding_acc=3, select_acc=0.001):
    pcfg, word_dict = load_model(path)
    if reduce:
        print(f"reducing PCFG probabilities to a rounding accuracy of {rounding_acc}")
        new_trans = reduce_transitions(pcfg[0], rounding_acc=rounding_acc, select_acc=select_acc)
        start_state = create_start_state(new_trans, pcfg[1], rounding_acc=rounding_acc, select_acc=select_acc)
    else:
        new_trans = pcfg[0]
        #new_trans = transform_transitions(pcfg[0])
        start_state = create_start_state_no_reduction(new_trans, pcfg[1])
    export_pcfg = translate_to_pcfg(new_trans, start_state, word_dict)
    return export_pcfg


def main():
    parser = start_parser()
    args = parser.parse_args()
    # create the corresponding select accuracy for our rounding accuracy
    select_acc = float(f"1e-{args.rounding_acc}")
    export_pcfg = pcfg_from_path(args.path, reduce=args.reduce,
                                 rounding_acc=args.rounding_acc, select_acc=select_acc)
    with open(args.outputfile, 'wb') as handle:
        pkl.dump(export_pcfg, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()