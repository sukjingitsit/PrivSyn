import json
import yaml
import numpy as np
import pandas as pd

class DataLoader():
    def __init__(self, datapath, configpath, datainfopath, dataschemapath):
        self.configpath = configpath
        with open(configpath, 'r', encoding="utf-8") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.dataschemapath = dataschemapath
        with open(dataschemapath, 'r', encoding="utf-8") as f:
            self.dataschema = json.load(f)
        self.general_schema = self.dataschema['schema']

        self.datainfopath = datainfopath
        with open(datainfopath, 'r', encoding="utf-8") as f:
            datainfopath = json.load(f)
        self.datatypes = datainfopath['dtype']
        self.fillna = datainfopath['fillna']

        self.public_data = None
        self.datapath = datapath
        self.private_data = pd.read_csv(datapath, dtype=self.datatypes)

        self.all_attrs = []

        self.encode_mapping = {}
        self.decode_mapping = {}

        self.pub_marginals = {}
        self.priv_marginals = {}
        self.priv_one_way = {}
        self.priv_two_way = {}
        self.priv_indif = {}
        self.encode_schema = {}
    
    def all_one_way_marginals (self, data):
        attributelist = self.all_attrs
        marginals = {}
        for attribute in attributelist:
            marginals[frozenset([attribute])] = self.one_way_marginal(attribute, data)
        return marginals
    
    def one_way_marginal(self, attribute, data):
        marginal = data.assign(n=1).pivot_table(values='n', index=attribute, aggfunc="sum", fill_value=0)
        indices = sorted([i for i in self.encode_mapping[attribute].values()])
        marginal = marginal.reindex(index=indices).fillna(0).astype(np.int32)
        if (data is self.private_data):
            self.priv_marginals[frozenset([attribute])] = self.priv_one_way[frozenset([attribute])] = marginal
        return marginal

    def all_two_way_marginals (self, data):
        attributelist = self.all_attrs
        marginals = {}
        for i, attribute in enumerate(attributelist):
            for j in range(i+1, len(attributelist)):
                marginals[frozenset([attribute, attributelist[j]])] = self.two_way_marginal(attribute, attributelist[j], data)
        return marginals
    
    def two_way_marginal(self, attributerow, attributecol, data):
        marginal = data.assign(n=1).pivot_table(values='n', index=[attributerow,attributecol],aggfunc="sum", fill_value=0)
        indexrow = sorted([i for i in self.encode_mapping[attributerow].values()])
        indexcol = sorted([i for i in self.encode_mapping[attributecol].values()])
        marginal = marginal.reindex(pd.MultiIndex.from_product([indexrow, indexcol], names = [attributerow, attributecol])).fillna(0).astype(np.int32)
        if (data is self.private_data):
            self.priv_marginals[frozenset([attributerow, attributecol])] = self.priv_two_way[frozenset([attributerow, attributecol])] = marginal
        return marginal

    def indif(self, attributerow, attributecol, data):
        n = len(data)
        marginalrow = self.one_way_marginal(attributerow, data)
        marginalcol = self.one_way_marginal(attributecol, data)
        marginalrowcol = self.two_way_marginal(attributerow, attributecol, data)
        indexrow = sorted([i for i in self.encode_mapping[attributerow].values()])
        indexcol = sorted([i for i in self.encode_mapping[attributecol].values()])
        indif = 0
        for i in range(len(marginalrow)):
            for j in range(len(marginalcol)):
                indif += np.abs(marginalrow['n'][i]*marginalcol['n'][j]/n - marginalrowcol.xs((indexrow[i], indexcol[j]))['n'])
        if (data is self.private_data):
            self.priv_marginals[frozenset([attributerow, attributecol])] = self.priv_two_way[frozenset([attributerow, attributecol])] = marginalrowcol
            self.priv_marginals[frozenset([attributerow])] = self.priv_one_way[frozenset([attributerow])] = marginalrow
            self.priv_marginals[frozenset([attributecol])] = self.priv_one_way[frozenset([attributecol])] = marginalcol
            self.priv_indif[frozenset([attributerow, attributecol])] = indif
        return indif
    
    def all_indifs(self, data):
        attributelist = self.all_attrs
        indifs = {}
        for i, attribute in enumerate(attributelist):
            for j in range(i+1, len(attributelist)):
                    indifs[frozenset([attribute, attributelist[j]])] = self.indif(attribute, attributelist[j], data)
        return indifs

    def data_loader(self):
        # Dealing with NaN values
        self.private_data.fillna(self.fillna, inplace=True, downcast = self.datatypes)
        # Remove identifier column
        self.private_data = self.remove_identifier(self.config['identifier_list'], self.private_data)
        # Applying binning to numeric types
        self.private_data = self.binning_attributes(self.config['binning_list'], self.private_data)
        # Applying grouping to attributes (Optional)
        # self.private_data = self.grouping_attributes(self.config['grouping_list'], self.private_data)
        # Encoding the remaining columns
        self.private_data = self.encode_remaining(self.general_schema, self.config, self.private_data)
        for attribute, encode_mapping in self.encode_mapping.items():
            self.encode_schema[attribute] = sorted(encode_mapping.values())
            self.all_attrs.append(attribute)
        display(self.private_data.head())

    def encode_remaining(self, schema, config, data):
        encoded_attributes = list(config['binning_list'].keys()) # + [grouping['grouped_name'] for grouping in config['grouping_list'] if grouping is not None]
        identifiers = self.config['identifier_list']
        for attribute in data.columns:    
            if attribute in identifiers or attribute in encoded_attributes:
                continue
            mapping = schema[attribute]['values']
            encoding = {v: i for i, v in enumerate(mapping)}
            data[attribute] = data[attribute].map(encoding)
            self.encode_mapping[attribute] = encoding
            self.decode_mapping[attribute] = mapping
        return data

    def grouping_attributes(self, grouping_list, data):
        for grouping in grouping_list:
            if grouping is None:
                return data
            grouped_attributes = grouping['attributes']
            new_attribute = grouping['grouped_name']
            data[new_attribute] = data[grouped_attributes].apply(tuple, axis=1)
            encoding = {v : i for i, v in enumerate(grouping['combinations'])}
            data[new_attribute] = data[new_attribute].map(encoding)
            self.encode_mapping[new_attribute] = encoding
            self.decode_mapping[new_attribute] = grouping['combinations']
            data = data.drop(grouped_attributes, axis=1)
        return data

    def binning_attributes(self, binning_info, data):
        for attr, spec_list in binning_info.items():
            [s, t, step] = spec_list
            bins = np.r_[-np.inf, np.arange(s, t, step), np.inf]
            data[attr] = pd.cut(data[attr], bins).cat.codes
            self.encode_mapping[attr] = {(bins[i], bins[i + 1]): i for i in range(len(bins) - 1)}
            self.decode_mapping[attr] = [i for i in range(len(bins) - 1)]
        return data

    def remove_identifier(self,identifier_list, data):
        for identifier in identifier_list:
            data = data.drop(identifier, axis=1)
        return data

if __name__ == '__main__':
    pass