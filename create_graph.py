import json

#superclasses = ["n02121620"]
# cat, dog, bird, car, truck
# vessel, lizard, snake, primate, beetle
superclasses = ["n02121620", "n02084071", "n01503061", "n02958343", "n04490091",
                "n04530566", "n01674464", "n01726692", "n02469914", "n02164464"]

FNAME = "wordnet.is_a.txt"
ID_TO_HUMAN_LABEL_FNAME = "words.txt"

if __name__ == "__main__":
    labels = {}
    graph = {}
    new_label_map = {}
    new_classes = {}

    with open(ID_TO_HUMAN_LABEL_FNAME, "r") as f:
        for line in f:
            id = line.split()[0]
            label = line[10:].strip()

            labels[id] = label


    with open(FNAME, "r") as f:
        for line in f:
            words = line.split()
            parent = words[0]
            child = words[1]

            if child not in graph:
                graph[child] = parent

    file = open('imagenet/classes.json', 'r')
    class_dict = json.load(file)


    # loop through every subclass in imagenet
    for subclass in class_dict.keys():
        #loop through every superclass
        for  superclass in superclasses:
            c = subclass
            while (graph.get(c) != None):
                if graph[c] == superclass:
                    new_label_map[subclass] = superclass
                    break
                c = graph.get(c)
            if subclass in new_label_map:
                break


    #print(labels[new_label_map["n02106030"]])

    new_classes = {}
    for s in superclasses:
        new_classes[s] = []

    print(len(new_classes))

    for key, val in new_label_map.items():
        #print(labels[val])
        #print(labels[key])
        new_classes[val].append(key)

        pass

    for l in new_classes.values():
        print(len(l))

    #print(labels["n02123785"])
    

 