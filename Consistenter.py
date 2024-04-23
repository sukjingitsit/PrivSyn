import copy
import numpy as np
import pandas as pd

class Consistenter():
    def __init__(self, anonymiser, attributelist):
        self.noisy_marginals = {}
        self.noise_marginals = {}
        self.noisy_one_way_marginals = {key : val.copy(deep = True) for key, val in anonymiser.noisy_one_way.items()}
        self.noisy_two_way_marginals = {key : val.copy(deep = True) for key, val in anonymiser.noisy_two_way.items()}
        self.noisy_multi_way_marginals = {key : val.copy(deep = True) for key, val in anonymiser.noisy_multi_way.items()}
        self.noise_one_way_marginals = anonymiser.noise_one_way
        self.noise_two_way_marginals = anonymiser.noise_two_way
        self.noise_multi_way_marginals = anonymiser.noise_multi_way
        self.attributelist = attributelist
        self.noisy_marginals.update(self.noisy_one_way_marginals)
        self.noisy_marginals.update(self.noisy_two_way_marginals)
        self.noisy_marginals.update(self.noisy_multi_way_marginals)
        self.noise_marginals.update(self.noise_one_way_marginals)
        self.noise_marginals.update(self.noise_two_way_marginals)
        self.noise_marginals.update(self.noise_multi_way_marginals)
        self.num_synthesize_records =  np.mean([np.sum(marginal['m']) for marginal in self.noisy_marginals.values()]).round().astype(int)
    
    def norm_sub_normalise(self, marginal):
        marginal[marginal['m'] < 0] = 0
        unnormalised_sum = np.sum(marginal['m'])
        marginal['m'] *= self.num_synthesize_records/unnormalised_sum
        #marginal = marginal.round().astype(int)
        return marginal

    def norm_sub_normalise_all(self, marginals):
        for marginal in marginals.values():
            self.norm_sub_normalise(marginal)
    
    def mean_estimate_one_way(self, marginals, noise, attributelist):
        for attribute in attributelist:
            marginaldict = {marginal : ((int) (marginals[marginal].size/marginals[frozenset([attribute])].size), noise[marginal]) for marginal in marginals.keys() if attribute in marginal}
            weights_unnormalised = [marginaldict[marginal][1]/marginaldict[marginal][0] for marginal in marginaldict.keys()]
            weights_normalised = {marginal : marginaldict[marginal][1]/marginaldict[marginal][0]/sum(weights_unnormalised) for marginal in marginaldict.keys()}
            estimated_marginal = self.noisy_one_way_marginals[frozenset([attribute])].copy(deep = True)
            estimated_marginal[estimated_marginal['m'] != 0] = 0
            for marginal in marginaldict.keys():
                if (len(marginal) == 1):
                    estimated_marginal['m'] += weights_normalised[marginal]*marginals[marginal]['m']
                else:
                    for val in estimated_marginal.index:
                        estimated_marginal.loc[val] += np.sum(marginals[marginal].xs(val, level = attribute), axis = 0)*weights_normalised[marginal]
            for marginal in marginaldict.keys():
                if (len(marginal) == 1):
                    marginals[marginal]['m'] = estimated_marginal['m']
                else:
                    for val in estimated_marginal.index:
                        tempmarginal = marginals[marginal].xs(val, level = attribute)
                        tempsum = np.sum(marginals[marginal].xs(val, level = attribute), axis = 0)
                        for tempval in tempmarginal.values:
                            tempval *= estimated_marginal.loc[val,'m']/tempsum
    
    def mean_estimate_two_way(self, marginals, noise):
        marginalslist = list(self.noisy_multi_way_marginals.keys())
        attpairs = set()
        for marginal1 in marginalslist:
            for marginal2 in marginalslist:
                if (marginal1 == marginal2):
                    continue
                else:
                    marginalintersection = marginal1.intersection(marginal2)
                    if (len(marginalintersection) == 2):
                        attpairs.add(marginalintersection)
        for attpair in attpairs:
            attpair = list(attpair)
            attpairmarginals = [marginal for marginal in marginalslist if attpair[0] in marginal and attpair[1] in marginal]
            marginaldict = {marginal : ((int) (marginals[marginal].size/(marginals[frozenset([attpair[0]])].size*marginals[frozenset([attpair[1]])].size)), noise[marginal]) for marginal in attpairmarginals}
            weights_unnormalised = [marginaldict[marginal][1]/marginaldict[marginal][0] for marginal in marginaldict.keys()]
            weights_normalised = {marginal : marginaldict[marginal][1]/marginaldict[marginal][0]/sum(weights_unnormalised) for marginal in marginaldict.keys()}
            estimated_marginal = {}
            for val1 in marginals[frozenset([attpair[0]])].index:
                for val2 in marginals[frozenset([attpair[1]])].index:
                    estimated_marginal[(val1, val2)] = 0
                    for marginal in marginaldict.keys():
                        estimated_marginal[(val1, val2)] += np.sum(marginals[marginal].xs((val1, val2), level = tuple(attpair)), axis = 0)*weights_normalised[marginal]
                    for marginal in marginaldict.keys():
                        tempmarginal = marginals[marginal].xs((val1, val2), level = tuple(attpair))
                        tempsum = np.sum(marginals[marginal].xs((val1, val2), level = tuple(attpair)), axis = 0)
                        for tempval in tempmarginal.values:
                            tempval *= estimated_marginal[(val1, val2)]/tempsum
    
    def make_consistent(self, iterations = 0):
        for i in range(iterations):
            self.norm_sub_normalise_all(self.noisy_marginals)
            self.mean_estimate_one_way(self.noisy_marginals, self.noise_marginals, self.attributelist)
            self.mean_estimate_two_way(self.noisy_marginals, self.noise_marginals)
            print(f"Iteration {i+1} of {iterations} to make the marginals consistent and normalise them")

if __name__ == '__main__':
    pass