from GraduallyUpdateMethod.View import View
from GraduallyUpdateMethod.ViewConsistenter import ViewConsistenter
from GraduallyUpdateMethod.RecordSynthesiser import RecordSynthesizer

import numpy as np
import pandas as pd
import numpy.linalg as LA

class GraduallyUpdateMethod():
    def __init__(self, dataloader, consistenter):
        self.synthesized_df = None
        self.update_iterations = None
        self.attrs_view_dict = {}
        self.onehot_view_dict = {}
        self.attr_list = []
        self.domain_list = []
        self.attr_index_map = {}
        self.dataloader = dataloader
        self.consistenter = consistenter

    def initialiser(self, view_iterations = 10):
        self.attr_list = self.dataloader.all_attrs
        self.domain_list = np.array([len(self.dataloader.encode_schema[att]) for att in self.attr_list])
        self.attr_index_map = {att: att_i for att_i, att in enumerate(self.attr_list)}
        noisy_onehot_view_dict, noisy_attr_view_dict = self.construct_views(self.consistenter.noisy_marginals)
        self.onehot_view_dict, self.attrs_view_dict = self.normalize_views(noisy_attr_view_dict, self.attr_index_map)
        viewconsistenter = ViewConsistenter(self.onehot_view_dict, self.domain_list, view_iterations)
        viewconsistenter.consist_views()
        for _, view in self.onehot_view_dict.items():
            view.count /= sum(view.count)
  
    def synthesize(self, iterations = 20, num_records = None):
        noisy_marginals = self.consistenter.noisy_marginals
        if num_records is not None:
            num_records = num_records
        else:
            num_records = self.consistenter.num_synthesize_records
        self.update_iterations = iterations
        print(f"num_rec: {num_records}")
        clusters = self.cluster(self.attrs_view_dict)
        attrs = self.dataloader.all_attrs
        print(f"attrs: {attrs}")
        domains = self.domain_list
        print(f"domains: {domains}")
        self.synthesize_records(attrs, domains, clusters, num_records)
        return self.synthesized_df

    def synthesize_records(self, attrs, domains, clusters, num_synthesize_records):
        for cluster_attrs, list_marginal_attrs in clusters.items():

            #singleton_views = {attr: self.attrs_view_dict[frozenset([attr])] for attr in attrs}
            
            singleton_views = {}
            for cur_attrs, view in self.attrs_view_dict.items():
                if len(cur_attrs) == 1:
                    singleton_views[cur_attrs] = view

            synthesizer = RecordSynthesizer(attrs, domains, num_synthesize_records)
            synthesizer.initialize_records(list_marginal_attrs, singleton_views=singleton_views)
            attrs_index_map = {attrs: index for index, attrs in enumerate(list_marginal_attrs)}

            for update_iteration in range(self.update_iterations):
                synthesizer.update_alpha(update_iteration)
                sorted_error_attrs = synthesizer.update_order(update_iteration, self.attrs_view_dict,
                                                              list_marginal_attrs)
                for attrs in sorted_error_attrs:
                    attrs_i = attrs_index_map[attrs]
                    synthesizer.prepare_update(self.attrs_view_dict[attrs])
                    synthesizer.update_records_prepare(self.attrs_view_dict[attrs])
                    synthesizer.update_records(self.attrs_view_dict[attrs], attrs_i)
                    
                print(f"Iteration {update_iteration+1} of {self.update_iterations} completed to generate the dataset using the noisy marginals")
            if self.synthesized_df is None:
                synthesizer.df = synthesizer.df.copy(deep = True)
                self.synthesized_df = synthesizer.df
            else:
                synthesizer.df = synthesizer.df.copy(deep = True)
                self.synthesized_df.loc[:, cluster_attrs] = synthesizer.df.loc[:, cluster_attrs]
            

    @staticmethod
    def normalize_views(noisy_view_dict, attr_index_map):
        views_dict = {}
        onehot_view_dict = {}
        for view_att, view in noisy_view_dict.items():
            views_dict[view_att] = view
            view_onehot = GraduallyUpdateMethod.one_hot(view_att, attr_index_map)
            onehot_view_dict[tuple(view_onehot)] = view
        return onehot_view_dict, views_dict

    @staticmethod
    def obtain_singleton_views(attrs_view_dict):
        singleton_views = {}
        for cur_attrs, view in attrs_view_dict.items():
            if len(cur_attrs) == 1:
                singleton_views[cur_attrs] = view
        return singleton_views


    def construct_views(self, marginals):
        onehot_view_dict = {}
        attr_view_dict = {}

        for marginal_att, marginal_value in marginals.items():
            view_onehot = GraduallyUpdateMethod.one_hot(marginal_att, self.attr_index_map)
            view = View(view_onehot, self.domain_list)
            view.count = marginal_value.values.flatten()
            onehot_view_dict[tuple(view_onehot)] = view
            attr_view_dict[marginal_att] = view


        return onehot_view_dict, attr_view_dict


    def log_result(self, result):
        self.d.append(result)

    @staticmethod
    def build_attr_set(attrs):
        attrs_set = set()

        for attr in attrs:
            attrs_set.update(attr)

        return tuple(attrs_set)

    def cluster(self, marginals):
        clusters = {}
        keys = []
        for marginal_attrs, _ in marginals.items():
            keys.append(marginal_attrs)

        clusters[GraduallyUpdateMethod.build_attr_set(marginals.keys())] = keys
        return clusters

    @staticmethod
    def one_hot(cur_att, attr_index_map):
        cur_view_key = [0] * len(attr_index_map)
        for attr in cur_att:
            cur_view_key[attr_index_map[attr]] = 1
        return cur_view_key

    @staticmethod
    def calculate_l1_errors(records, target_marginals, attrs_view_dict):
        l1_T_Ms = []
        l1_T_Ss = []
        l1_M_Ss = []

        for cur_attrs, target_marginal_pd in target_marginals.items():
            view = attrs_view_dict[cur_attrs]
            syn_marginal = view.count_records_general(records)
            target_marginal = target_marginal_pd.values.flatten()

            T = target_marginal / np.sum(target_marginal)
            M = view.count
            S = syn_marginal / np.sum(syn_marginal)

            l1_T_Ms.append(LA.norm(T - M, 1))
            l1_T_Ss.append(LA.norm(T - S, 1))
            l1_M_Ss.append(LA.norm(M - S, 1))

        return np.mean(l1_T_Ms), np.mean(l1_T_Ss), np.mean(l1_M_Ss)

if __name__ == '__main__':
    pass