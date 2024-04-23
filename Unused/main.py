from DataLoader import DataLoader
from Anonymisation import Anonymisation
from Consistenter import Consistenter
from GUM import GraduallyUpdateMethod
from PostProcessor import RecordPostprocessor
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type = str, default = "UCIMLAdult/uciml_adult.csv", help = "Path to CSV file containing public data records")
parser.add_argument("--config_path", type = str, default = "UCIMLAdult/data.yaml", help = "Path to yaml file containing data configuration information")
parser.add_argument("--info_path", type = str, default = "UCIMLAdult/column_info.json", help = "Path to json file containing column information")
parser.add_argument("--schema_path", type = str, default = "UCIMLAdult/loading_data.json", help = "Path to json file containing data schema")
parser.add_argument("--epsilon", type = float, default = 10)
parser.add_argument("--delta", type = float, default = 3e-11)
parser.add_argument("--num_records", type = int, default = 0)

args = parser.parse_args()
data_path = args.data_path
config_path = args.config_path
info_path = args.info_path
schema_path = args.schema_path
epsilon = args.epsilon
delta = args.delta
num_records = args.num_records

dl = DataLoader(data_path, config_path, info_path, schema_path)
dl.data_loader()
dl.all_indifs(dl.private_data)

anon = Anonymisation(10,3e-11)
anon.anonymiser(dl)

c = Consistenter(anon, dl.all_attrs)
c.make_consistent(iterations = 5)

gum = GraduallyUpdateMethod(dl, c)
gum.initialiser(view_iterations = 100)
syn_data = gum.synthesize(iterations = 50, num_records = int(c.num_synthesize_records) if num_records == 0 else num_records)

processor_priv = RecordPostprocessor(dl.private_data, dl.configpath, dl.datainfopath, dl.decode_mapping)
processor_pub = RecordPostprocessor(syn_data, dl.configpath, dl.datainfopath, dl.decode_mapping)
priv_data = processor_priv.post_process()
pub_data = processor_pub.post_process()

t = 0
for i in range(len(dl.all_attrs)):
    for j in range(i+1, len(dl.all_attrs)):
        attr1 = dl.all_attrs[i]
        attr2 = dl.all_attrs[j]
        priv_marg = dl.two_way_marginal(attr1, attr2, dl.private_data)/len(dl.private_data)
        pub_marg = dl.two_way_marginal(attr1, attr2, syn_data)/len(syn_data)
        t += (priv_marg - pub_marg).abs().sum().sum()
print("Marginal Difference Score :", t/(len(dl.all_attrs)*(len(dl.all_attrs)-1)))