import copy
import numpy as np
import pandas as pd
import networkx as nx

class Anonymisation():
    def __init__(self, epsilon, delta):
        self.epsilon = epsilon
        self.delta = delta
        self.rho = (np.sqrt((-np.log(delta)) + epsilon) - np.sqrt(-np.log(delta)))**2
        print(f"epsilon: {epsilon}, delta: {delta}, rho: {self.rho}")
        self.noisy_one_way = {}
        self.noisy_multi_way = {}
        self.noisy_two_way = {}
        self.noise_one_way = {}
        self.noise_two_way = {}
        self.noise_two_way = {}
        self.noise_multi_way = {}
        self.noisy_indifs = {}
        self.selected_two_ways_list = None
        self.selected_two_way_graph = None
        self.multi_marginals_list = None
        self.removed_marginals_list = None
        self.multi_marginals = {}
        self.total_domain_size = None

    def gaussian_mechanism(self, marginals, chosenmarginallist = None, removedmarginallist = None, frac = 0.1, typekey = "one-way"):
        rho_net = frac*self.rho
        noisy_marginals = {}
        noise_marginals = {}
        if (typekey == "one-way"):
            sigma = np.sqrt(len(marginals)/(2*rho_net))
            # sigma = np.sqrt(1/(2*rho_net))
            print(f"one-way sigma: {sigma}")

            for attribute, marginal in marginals.items():
                numvals = len(marginal)
                noise = np.random.normal(loc = 0, scale = sigma, size = numvals)
                marginal['m'] = np.add(marginal['n'],noise)
                noisy_marginals[attribute] = marginal.copy(deep = True).drop('n', axis = 1)
                noise_marginals[attribute] = rho_net/len(marginals)
                # The code to check for low count thresholds hasn't been implemented yet.
            self.noisy_one_way = noisy_marginals
            self.noise_one_way = noise_marginals
        if (typekey == "two-way"):
            sumctp = 0
            for attpair in chosenmarginallist:
                if attpair in removedmarginallist:
                    continue
                sumctp += pow(len(marginals[frozenset(attpair)]),2/3)
            for attpair in chosenmarginallist:
                if attpair in removedmarginallist:
                    continue
                numvals = len(marginals[frozenset(attpair)])
                rho_val = rho_net*pow(len(marginals[frozenset(attpair)]),2/3)/sumctp
                sigma = np.sqrt(1/(2*rho_val))
                print(f"sigma {attpair}: {sigma}")
                noise = np.random.normal(loc = 0, scale = sigma, size = numvals)
                marginals[frozenset(attpair)]['m'] = np.add(marginals[frozenset(attpair)]['n'],noise)
                noisy_marginals[frozenset(attpair)] = marginals[frozenset(attpair)].copy(deep = True).drop('n', axis = 1)
                noise_marginals[frozenset(attpair)] = rho_val
            self.noisy_two_way = noisy_marginals
            self.noise_two_way = noise_marginals
        if (typekey == "multi-way"):
            dictmarginals = {}
            for marginal in marginals:
                if len(marginal) == 2 and marginal in self.selected_two_ways_list and marginal not in self.removed_marginals_list:
                    dictmarginals[marginal] = marginals[marginal]
                if len(marginal) > 2 and marginal in self.multi_marginals:
                    dictmarginals[marginal] = marginals[marginal]
            sigma = np.sqrt(len(dictmarginals)/(2*rho_net))
            # sigma = np.sqrt(1/(2*rho_net))
            print(f"multi-way sigma: {sigma}")
            for attributes, marginal in dictmarginals.items():
                numvals = len(marginal)
                noise = np.random.normal(loc = 0, scale = sigma, size = numvals)
                marginal['m'] = np.add(marginal['n'],noise)
                noisy_marginals[attributes] = marginal.copy(deep = True).drop('n', axis = 1)
                noise_marginals[attributes] = rho_net/len(dictmarginals)
            self.noisy_multi_way = noisy_marginals
            self.noise_multi_way = noise_marginals
    
    def get_noisy_indifs(self, indifs, rho):
        numattpairs = len(indifs)
        sigma = numattpairs*rho
        print(f"indif sigma: {sigma}")
        noisy_indifs = {}
        for attpair, indifval in indifs.items():
            noisy_indifs[attpair] = indifval + np.random.normal(0,sigma)
        self.noisy_indifs = noisy_indifs
        return noisy_indifs

    def two_way_selection(self, two_way_marginals, indifs, frac = 0.1):
        rho_net = frac*self.rho
        marginalsize = {}
        chosenset = set()
        remainset = set()
        for attpair, marginal in two_way_marginals.items():
            marginalsize[attpair] = marginal.size
            remainset.add(attpair)
        noisy_indifs = self.get_noisy_indifs(indifs, rho_net) # rho' < rho, but by how much was never defined
        sumcpt = 0
        preveval_minval = None
        while True:
            eval_dict = {}
            for attpair in remainset:
                sumcpt_temp = sumcpt + pow(marginalsize[attpair],2/3)
                eval = pow(marginalsize[attpair],2/3)*np.sqrt(sumcpt_temp/np.pi)
                for chosen_attpair in chosenset:
                    eval += pow(marginalsize[chosen_attpair],2/3)*np.sqrt(sumcpt_temp/np.pi)
                for unchosen_attpair in remainset:
                    if unchosen_attpair == attpair:
                        continue
                    else:
                        eval += noisy_indifs[unchosen_attpair]
                eval_dict[attpair] = eval
            if len(chosenset) == len(two_way_marginals):
                self.selected_two_ways_list = list(chosenset)
                break
            eval_minval = min(eval_dict.values())
            eval_minattpair = None
            for attpair in eval_dict.keys():
                if (eval_dict[attpair] == eval_minval):
                    eval_minattpair = attpair
                    break
                else:
                    continue
            if len(chosenset) == 0:
                chosenset.add(eval_minattpair)
                remainset.remove(eval_minattpair)
                sumcpt += pow(marginalsize[eval_minattpair],2/3)
            elif preveval_minval < eval_minval:
                self.selected_two_ways_list = list(chosenset)
                break
            else:
                chosenset.add(eval_minattpair)
                remainset.remove(eval_minattpair)
                sumcpt += pow(marginalsize[eval_minattpair],2/3)
            preveval_minval = eval_minval
        return chosenset
    
    def two_way_selected_graph (self, selected_two_ways, attributelist):
        selected_two_way_graph = nx.Graph()
        for attribute in attributelist:
            selected_two_way_graph.add_node(attribute)
        for pair in selected_two_ways:
            pairlist = list(pair)
            selected_two_way_graph.add_edge(pairlist[0], pairlist[1])
            selected_two_way_graph.add_edge(pairlist[1], pairlist[0])
        self.selected_two_way_graph = selected_two_way_graph
        return selected_two_way_graph
    
    def select_multi_marginals(self, selected_two_way_graph, attributelist):
        combined_attributes = set()
        multi_marginals_list = []
        removed_marginals_list = set()
        cliques = list(nx.enumerate_all_cliques(selected_two_way_graph))
        self.total_domain_size = 1
        for attribute in attributelist:
            self.total_domain_size *= len(self.noisy_one_way[frozenset([attribute])])
        for clustersize in range(len(attributelist),2,-1):
            chosencliques  = [clique for clique in cliques if len(clique) == clustersize]
            for clique in chosencliques:
                domain_size = 1
                for marginal in clique:
                    domain_size *= len(self.noisy_one_way[frozenset([marginal])])
                if len(combined_attributes.intersection(clique)) <= 2 and domain_size < pow(self.total_domain_size, 1/4):
                    combined_attributes = combined_attributes.union(clique)
                    multi_marginals_list.append(frozenset(clique))
                    for i in range(len(clique)):
                        for j in range(i+1,len(clique)):
                            removed_marginals_list.add(frozenset([clique[i], clique[j]]))
        self.multi_marginals_list = multi_marginals_list
        self.removed_marginals_list = list(removed_marginals_list)
        return multi_marginals_list, list(removed_marginals_list)

    def generate_multi_marginals(self, selected_marginals, dataloader):
        for marginal in selected_marginals:
            if np.prod([len(dataloader.priv_one_way[frozenset([i])]) for i in marginal]) > np.sqrt(np.prod([len(dataloader.priv_one_way[frozenset([i])]) for i in dataloader.all_attrs])):
                continue
            attributes = [attribute for attribute in marginal]
            marginal = dataloader.private_data.assign(n=1).pivot_table(values='n', index=attributes,aggfunc="sum", fill_value=0)
            indices = [sorted([i for i in dataloader.encode_mapping[attribute].values()]) for attribute in attributes]
            marginal = marginal.reindex(pd.MultiIndex.from_product(indices, names = attributes)).fillna(0).astype(np.int32)
            dataloader.priv_marginals[frozenset(attributes)] = marginal
            self.multi_marginals[frozenset(attributes)] = marginal
    
    def anonymiser(self, dataloader):
        self.gaussian_mechanism(dataloader.priv_one_way)
        self.two_way_selection(dataloader.priv_two_way, dataloader.priv_indif)
        self.two_way_selected_graph(self.selected_two_ways_list, dataloader.all_attrs)
        self.select_multi_marginals(self.selected_two_way_graph, dataloader.all_attrs)
        self.generate_multi_marginals(self.multi_marginals_list, dataloader)
        self.gaussian_mechanism(dataloader.priv_two_way, self.selected_two_ways_list, self.removed_marginals_list, typekey="two-way")
        self.gaussian_mechanism(dataloader.priv_marginals, self.selected_two_ways_list, self.removed_marginals_list, frac = 0.8, typekey="multi-way")