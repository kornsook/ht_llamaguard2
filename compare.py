from datasets import Dataset
from tqdm import tqdm
import argparse
from multiprocessing import Process, Manager
import math
import pandas as pd

def sub_compute(vals1, vals2, exact_match_lst, mutual_match_lst):
    exact_match, mutual_match = 0,0
    for i in tqdm(range(len(vals1))):
        val1 = vals1[i]
        val2 = vals2[i]
        # print(i)
        exact_match += int(val1 == val2)
        if val1 not in ("safe", "undefined") and val2 not in ("safe", "undefined"):
            lst1 = val1.split(",")
            lst2 = set(val2.split(","))
            for val in lst1:
                if val in lst2:
                    mutual_match += 1
                    break
        else:
            mutual_match += int(val1 == val2)
    # print(exact_match)
    # print(mutual_match)
    exact_match_lst.append(exact_match)
    mutual_match_lst.append(mutual_match)

parser = argparse.ArgumentParser(description="Your Script Description")

#     parser.add_argument("folder_name", type=str, help="Name of the folder")
parser.add_argument("--file1", type=str, help="File 1's path")
parser.add_argument("--file2", type=str, help="File 2's path")
parser.add_argument("--n-processors", type=int, help="# of processors")

args = parser.parse_args()

data1 = Dataset.from_json(args.file1).to_pandas()
data2 = Dataset.from_json(args.file2).to_pandas()
if data1['post_id'].isna().all():
    n = len(data1)
    vals1 = list(data1['llama_guard2_label'])
    vals2 = list(data2['llama_guard2_label'])
else:
    merged_df = pd.merge(data1[['post_id', 'llama_guard2_label']], data2[['post_id', 'llama_guard2_label']], on='post_id', how='inner').reset_index(drop=True)
    n = len(merged_df)
    vals1 = list(merged_df['llama_guard2_label_x'])
    vals2 = list(merged_df['llama_guard2_label_y'])
processes = []
# exact_match_lst = []
# mutual_match_lst = []
manager = Manager()
exact_match_lst = manager.list()
mutual_match_lst = manager.list()

idx = 0
# print(len(vals1))
# print(len(vals2))
for i in range(args.n_processors):
    batch_size = math.ceil((n - idx) / (args.n_processors - i))
    st = idx
    ed = min(n, idx + batch_size)
    # print(vals1[st:ed])
    p = Process(target = sub_compute, args=(vals1[st:ed], vals2[st:ed], exact_match_lst, mutual_match_lst))
    p.start()
    processes.append(p)
    idx = ed
for p in processes:
    p.join()

print(f"Size: {n}")
print(f"Exact match: {sum(exact_match_lst) / n}")
print(f"Mutual match: {sum(mutual_match_lst) / n}")