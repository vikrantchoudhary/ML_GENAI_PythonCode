# generator data & process them in chunks , do not load all data at once.

def batch_generator (file_path,batch_size=32):
    with open(file_path,'r') as f:
        batch = []
        for line in f:
            batch.append(line.strip())
            if (len(batch) == batch_size):
                yield batch #pause and return the batch
                batch=[]
        if batch:
            yield batch

#example
epochs =5
for epoch in range(epochs) :
    for batch in batch_generator("order_items.csv"):
        print(batch)
        print("\n===== \n")
#        model.train_on_batch(batch)
