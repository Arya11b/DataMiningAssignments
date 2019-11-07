import efficient_apriori as ep
from apriori import apriori as apriori
transactions = [
    ('A', 'B', 'D', 'G')
    , ('B', 'D', 'E')
    , ('A', 'B', 'C', 'E', 'F')
    , ('B', 'D', 'E', 'G')
    , ('A', 'B', 'C', 'E', 'F')
    , ('B', 'E', 'G')
    , ('A', 'C', 'D', 'E')
    , ('B', 'E')
    , ('A', 'B', 'E', 'F')
    , ('A', 'C', 'D', 'E')
]
# rules
itemsets,rules = ep.apriori(transactions=transactions,min_support=0.4)
print(rules)
# section a & b
print('a. b.')
apriori(transactions=transactions,support_threshold=4)
# section c
print('c.')
apriori(transactions=transactions,support_threshold=7)
# section d
print('d.')
itemsets2,rules2 = ep.apriori(transactions=transactions,min_support=0.4,min_confidence=1)
print(rules2)
# section e
print('e.')
rules_rhs = filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
  print(rule) # Prints the rule and its confidence, support, lift, ...