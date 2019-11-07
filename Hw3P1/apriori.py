import itertools as it
def apriori(transactions,support_threshold):
    items = []
    for t in transactions:
        for item in t:
            if item not in items:
                items.append(item)
    k = 1
    while( k <= len(items)):
        comb = list(it.combinations(items,k))
        count = [0 for i in range(len(comb))]
        for t in transactions:
            for c in comb:
                if set(c).issubset(t):
                    count[comb.index(c)] += 1
        i = 0
        while i < len(count):
            if count[i] < support_threshold:
                comb.pop(i)
                count.pop(i)
            else:
                i +=1
        print('frequent ' + str(k) + ' itemsets: ')
        print(comb)
        print(count)
        items = []
        for x in comb:
            for y in x:
                if y not in items:
                    items.append(y)
        k += 1
