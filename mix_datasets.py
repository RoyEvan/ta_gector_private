import random

# load data
with open("datasets/fce_train_train_bea19_os.txt", encoding="utf-8") as f:
    fce = f.readlines()

with open("datasets/agentlans/train_preprocessed_utf8.txt", encoding="utf-8") as f:
    agent = f.readlines()

# shuffle
random.shuffle(fce)
random.shuffle(agent)

# ambil sesuai rasio
fce_part = fce[:int(len(fce)*0.7)]
agent_part = agent[:int(len(fce)*0.3)]  # penting: base dari FCE size

# gabung
mix = fce_part + agent_part
random.shuffle(mix)

# save
with open("datasets/mix_datasets/mix_train.txt", "w", encoding="utf-8") as f:
    f.writelines(mix)

print("DONE")