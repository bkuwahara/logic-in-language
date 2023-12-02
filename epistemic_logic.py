import re


class Modal:
    def __init__(self, agent, proposition):
        self.agent = agent
        self.proposition = proposition
        
    def to_set(self):
        modals = {self}
        if isinstance(self.proposition, Conjunction):
            modals.remove(self)
            modals = modals.union({type(self)(self.agent, clause) for clause in self.proposition.to_set()})
        elif isinstance(self.proposition, Modal):
            modals.remove(self)
            modals = modals.union({type(self)(self.agent, clause) for clause in self.proposition.to_set()})
        return modals
    
    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)
        

class BeliefOperator(Modal):
    def __repr__(self):
        return f"B({self.agent},{self.proposition})"

        
class KnowledgeOperator(Modal):
    def __repr__(self):
        return f"K({self.agent},{self.proposition})"

                
class Conjunction:
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("Conjunction must have at least two clauses.")
        self.clauses = set(args)
        
    def __repr__(self):
        return " ^ ".join([str(clause) for clause in self.clauses])
    
    def to_set(self):
        output = set()
        for clause in self.clauses:
            if isinstance(clause, Modal) or isinstance(clause, Conjunction):
                output = output.union(clause.to_set())
            else:
                output.add(clause)
                
        return output
    




# Finds all independent (non-nested) modal operators in an input string
# Returns as a tuple of (operator_type, agent, proposition)
def get_modals(input_string):
    operators = {"K" : KnowledgeOperator, "B" : BeliefOperator}
    output = []
    
    _exprs = re.findall('[KB]\(([^)]+)', input_string)
    if not _exprs:
        return [input_string]
    
    exprs = [x + ')'*x.count('(') for x in _exprs]
    for expr in exprs:
        agent, prop = expr.split(',', maxsplit=1)
        idx = input_string.find(f'({expr})')
        op = operators[input_string[idx-1]]
        
        # Recurse into proposition to get its structure (if any)
        prop_structure = get_modals(prop)
        if len(prop_structure) < 2:
            output.append(op(agent, prop_structure[0]))
        else:
            output.append(op(agent, Conjunction(*prop_structure)))
        
    return output


def to_logic(input_string):
    modals = get_modals(input_string)
    if len(modals) < 2:
        return modals[0]
    else:
        return Conjunction(*modals)
    

    
def to_belief(formula):
    if isinstance(formula, str):
        return formula
    elif isinstance(formula, Conjunction):
        return Conjunction(*[to_belief(p) for p in formula.clauses])
    elif isinstance(formula, BeliefOperator):
        return BeliefOperator(formula.agent, to_belief(formula.proposition))
    elif isinstance(formula, KnowledgeOperator):
        internal = to_belief(formula.proposition)
        return Conjunction(BeliefOperator(formula.agent, internal), internal)
    else:
        raise ValueError("formula must be a string or logical type (Conjunction, KnowledgeOperator, or BeliefOperator)")
            
        return output
                


class KnowledgeBase:
    def extract_agents(formula):
        agents = set()
        if hasattr(formula, "agent"):
            agents.add(formula.agent)
            if hasattr(formula, "proposition"):
                agents = agents.union(KnowledgeBase.extract_agents(formula.proposition))
        return agents
                    
    def __init__(self, formulas):
        self.formulas = set()
        agents = set()
        for formula in formulas:
            formula = to_belief(formula)
            agents = agents.union(KnowledgeBase.extract_agents(formula))
            self.formulas = self.formulas.union(formula.to_set())
        self.agents = agents
        
        
    def from_string(input_string):
        return KnowledgeBase(to_logic(input_string).to_set())
    
    def entails(self, hypothesis):
        if isinstance(hypothesis, KnowledgeBase):
            return hypothesis.formulas.issubset(self.formulas)
        elif isinstance(hypothesis, Modal) or isinstance(hypothesis, Conjunction):
            return hypothesis.to_set().issubset(self.formulas)
        elif isinstance(hypothesis, str):
            return hypothesis in self.formulas
        return hypothesis.issubset(self.formulas)
    