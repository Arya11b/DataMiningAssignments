import efficient_apriori as ep
from apriori import apriori as apriori
transactions = [
    ('A', 'C', 'D')
    , ('B', 'C', 'E')
    , ('A', 'B', 'C', 'E')
    , ('B', 'E')
]
#rules
print('rules')
itemset, rules = ep.apriori(transactions=transactions,min_confidence=0.65,min_support=0.5)
print(rules)
# section a
print('.a')
apriori(transactions=transactions,support_threshold=2)
# section b
print('.b')
itemset2, rules2 = ep.apriori(transactions=transactions,min_confidence=0.65,min_support=0.5)
print(rules2)
# section c
print('.c')
itemset3, rules3 = ep.apriori(transactions=transactions,min_confidence=0.8,min_support=0.5)
print(rules3)
# section d
print('.d')
rules_rhs = filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
  print(rule) # Prints the rule and its confidence, support, lift, ...