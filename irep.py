import getopt
import math
import sys
import random

#
# Global META variable.
#
meta = {
    # The options dictionary contains all the user defined arguments.
    'opts': {},
    # The attribute dictionary contains all the attributes found in the attribute file.
    'attrs': {},
    # Table storing all the data except defining attribute.
    'data': [],
}


def main():
    """Main execution method."""
    # Determine command line arguments
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "e:a:c:t:m:o:")
    except getopt.GetoptError as err:
        print(err)
        help()
        sys.exit(2)

    # Process each command line argument.
    for o, a in opts:
        meta['opts'][o[1]] = a

    # The following arguments are required in all cases.
    for opt in ['e', 'a', 'c', 't', 'm', 'o']:
        if opt not in meta['opts']:
            help()
            sys.exit(2)

    # Execute option can be only one of two types.
    if meta['opts']['e'] != 'learn' and meta['opts']['e'] != 'classify':
        help()
        sys.exit(2)

    # Create attrs.
    f = open(meta['opts']['a'])
    meta['attrs'] = create_attrs(f)
    f.close()

    # Create classes and parse data.
    f = open(meta['opts']['t'])
    classes, meta['data'] = create_classes_and_data(f)
    f.close()

    if meta['opts']['e'] == 'learn':
        learn(classes)
    else:
        classify(classes)


def classify(classes):
    """
    Classifies given test cases.

    Key arguments:
    classes -- the testing cases to classify.
    """
    f = open(meta['opts']['m'], 'r')
    model = eval(f.readline())
    f.close()

    # Default class is the only one not found in the model.
    default = set(classes.keys()).difference(model.keys()).pop()

    # Keep track of the results for each case.
    results = {default: (len(classes[default]), 0)}

    # Sort in order
    items = classes.items()
    items.sort(key=lambda i: len(i[1]))

    # Loop over every test case and attempt to classify.
    for c, cases in items:

        # Init the results for this class.
        results[c] = (len(cases), 0)

        for case in cases:

            # This case was satisfied by non-default classes.
            found = False

            for m, ruleset in model.iteritems():

                # No need to continue of we found a rule that satisfies the case.
                if found:
                    break

                for rule in ruleset:
                    if bindings([case], [rule]):
                        if c == m:
                            results[c] = (results[c][0], results[c][1] + 1)
                        found = True
                        break

            # The case wasn't satisfied by any of the ruleset.
            # Check if it was our default case.
            if not found and c == default:
                results[default] = (results[default][0], results[default][1] + 1)

    total = 0
    classified = 0

    # Output our results to the result file.
    output = "Class\t\t\tCases\t\t\tClassified\n\n"
    for attr, result in results.iteritems():
        total += result[0]
        classified += result[1]
        output += attr + "\t\t\t" + str(result[0]) + "\t\t\t" + str(result[1]) + "\n"
    output += "\n"
    output += "Accuracy: " + str(classified * 100 / float(total)) + "%"
    f = open(meta['opts']['o'], 'w')
    f.write(output)
    f.close()

    # Dirty trick to capture output for our corrupter.
    print(str(classified * 100 / float(total)))


def learn(classes):
    """
    Learns the classes and outputs to model and output file.

    Key arguments:
    classes -- the two classes to learn.
    """
    # Sort our classes in order of least prevalence.  Because python works
    #  with lists backwards, it is going to run in reverse
    items = classes.items()
    items.sort(key=lambda i: len(i[1]), reverse=True)

    # The rule set that is used to write to file.
    output_ruleset = {}

    # Clean output
    output = ""

    # The positive training set consists of the less numbered classes cases.
    pos_attr = items[1][0]
    pos = items[1][1]

    # The negative training set is built from the remaining class.
    neg_attr = items[0][0]
    neg = items[0][1]

    # Build the ruleset.
    ruleset = irep(pos, neg)

    output_ruleset[pos_attr] = ruleset

    str_ruleset = ""
    for rule in ruleset:
        str_rule = ""
        for attr, value in rule.iteritems():
            str_rule += str(attr) + " " + str(value[0]) + " " + str(value[1]) + " && "
        str_ruleset += str_rule[:-4] + " || "

    output += "IF " + str_ruleset[:-4] + " THEN " + str(pos_attr) + "\n"

    # Write the model to file
    f = open(meta['opts']['m'], 'w')
    f.write(str(output_ruleset))
    f.close()

    output += "ELSE " + neg_attr

    # Write the model to file
    f = open(meta['opts']['o'], 'w')
    f.write(str(output))
    f.close()


def irep(pos, neg):
    """
    IREP++

    Key arguments:
    pos   -- the positive training cases id.
    neg   -- the negative training cases id.
    """
    # The final rule set.
    rule_set = []

    # Bad rule counter.
    counter = 0

    while pos:
        pos_len = len(pos)
        neg_len = len(neg)

        # Growing to pruning set ratio.
        ratio = 2 / float(3)

        pos_chunk = int(pos_len * ratio)
        neg_chunk = int(neg_len * ratio)

        if pos_chunk == 0:
            pos_chunk = 1

        if neg_len > 1 and neg_chunk == 0:
            neg_chunk = 1

        # Randomize sets.
        random.shuffle(pos)
        random.shuffle(neg)

        # Grow.
        grow_pos = pos[:pos_chunk]
        grow_neg = neg[:neg_chunk]
        rule = grow_rule(grow_pos, grow_neg)

        # Prune.
        prune_pos = pos[pos_chunk:]
        prune_neg = neg[neg_chunk:]
        rule = prune_rule(prune_pos, prune_neg, rule)

        #  Checking if rule is good.
        if count_cases(pos, rule) >= count_cases(neg, rule):
            # Add our rule to the ruleset.
            rule_set.append(rule)

            # Reset bad rule counter.
            counter = 0

            # Remove the cases found by rule.
            pos = remove_cases(pos, rule)
            neg = remove_cases(neg, rule)
        else:
            # Increment bad rule counter.
            counter += 1

            # If 5 bad rules have been grown, algorithm ends.
            if counter >= 5:
                return rule_set

    return rule_set


def grow_rule(pos, neg):
    """
    Grows a rule until it matches no negative cases.

    Key arguments
    pos   -- the positive training cases id.
    neg   -- the negative training cases id.
    """
    rule = {}
    rules = []
    attr_tables = {}

    # Creates attribute tables containing feature values and id of case from which the value comes.
    for case_id in pos + neg:
        for attr, value in meta['data'][case_id].items():
            if attr in attr_tables:
                attr_tables[attr].append((value, case_id))
            else:
                attr_tables[attr] = [(value, case_id)]

    # Sorting attribute tables.
    for attr in attr_tables:
        attr_tables[attr] = sorted(attr_tables[attr])

    while True:
        max_gain = None
        max_condition = None

        for attr in attr_tables.items():

            # Can't add an attribute twice.
            if attr[0] in rule:
                continue

            # Conditions for this attribute.
            conditions = []

            for v in attr[1][:]:
                conditions.append((attr[0], (">=", v[0])))
                conditions.append((attr[0], ("<=", v[0])))

            # Check the gain for each condition.
            for condition in conditions:
                gain = foil(pos, neg, condition, rule, rules)
                if max_gain < gain:
                    max_condition = condition
                    max_gain = gain

        # Add the new max condition.
        rule[max_condition[0]] = max_condition[1]
        rules.append(rule)

        # Remove covered cases from training set.
        pos, removed_pos = remove_cases_get_removed(pos, rules)
        neg, removed_neg = remove_cases_get_removed(neg, rules)

        # Remove covered cases from feature tables.
        for case in removed_pos + removed_neg:
            for attr in attr_tables.items():
                for v in attr[1]:
                    if v[1] is case:
                        attr[1].remove(v)

        # Check if it covers no negative cases.
        if max_gain <= 0.0 or bindings(neg, rules) == 0:
            rules.pop()
            return rule
        else:
            rules.pop()


def prune_rule(pos, neg, rule):
    """
    Prunes a rule.

    Key arguments:
    pos   -- the positive training cases id.
    neg   -- the negative training cases id.
    rule  -- the rule to prune.
    """

    # Deep copy our rule.
    tmp_rule = dict(rule)

    p = bindings(pos, [rule])
    n = bindings(neg, [rule])

    if p == 0 and n == 0:
        return tmp_rule

    max_rule = dict(tmp_rule)
    max_score = 0

    keys = max_rule.keys()
    i = -1

    while len(tmp_rule.keys()) > 1:
        # Remove the last attribute.

        lit = (keys[i], tmp_rule[keys[i]])

        del tmp_rule[keys[i]]

        # Recalculate score.
        p = bindings(pos, [rule])
        n = bindings(neg, [rule])

        if p == 0 and n == 0:
            continue

        tmp_score = foil(pos, neg, lit, tmp_rule, [])

        # If score is negative then there is no use in adding last condition.
        if tmp_score < max_score:
            max_rule = dict(tmp_rule)
            max_score = tmp_score

        i -= 1

    return max_rule


def foil(pos, neg, lit, rule, rules):
    """
    FOIL-GAIN

    Key arguments:
    pos   -- the positive cases.
    neg   -- the negative cases.
    lit   -- the literal to add.
    rule  -- the rule.
    rules -- the rules to compare to.
    """
    # Append our rule to the rule list.
    rules.append(rule)

    if rule:
        p0 = bindings(pos, rules)
        n0 = bindings(neg, rules)
    else:
        p0 = len(pos)
        n0 = len(neg)

    # Remove rule from stack.
    rules.pop()

    # Make a new rule with the literal.
    new_rule = dict(rule)
    new_rule[lit[0]] = lit[1]

    # Append the new rule and test bindings.
    rules.append(new_rule)

    p1 = bindings(pos, rules)
    n1 = bindings(neg, rules)

    # Remove new rule from stack.
    rules.pop()

    # Check for division by 0.
    if p0 == 0:
        d0 = 0
    else:
        d0 = math.log(float(p0) / (float(p0) + float(n0)), 2)

    if p1 == 0:
        d1 = 0
    else:
        d1 = math.log(float(p1) / (float(p1) + float(n1)), 2)

    # Pop both rules on stack.
    rules.append(rule)
    rules.append(new_rule)

    t = bindings(pos, rules)

    # Pop both rules off of stack.
    rules.pop()
    rules.pop()

    return t * (d1 - d0)


def count_cases(cases, rule):
    """
    Counts cases based on a rule.

    Key arguments:
    cases -- the cases to analyze.
    rule  -- the rule to analyze with.
    """
    count = 0
    for case in cases:
        if bindings([case], [rule]):
            count += 1

    return count


def remove_cases(cases, rule):
    """
    Removes cases based on a rule.

    Key arguments:
    cases -- the cases to analyze.
    rule  -- the rule to analyze with.
    """
    new_cases = []
    for case in cases:
        if not bindings([case], [rule]):
            new_cases.append(case)

    return new_cases


def remove_cases_get_removed(cases, rule):
    """
    Removes cases based on a rule.

    Key arguments:
    cases -- the cases to analyze.
    rule  -- the rule to analyze with.
    """
    new_cases = []
    removed_cases = []
    for case in cases:
        if not bindings([case], rule):
            new_cases.append(case)
        else:
            removed_cases.append(case)

    return new_cases, removed_cases


def bindings(cases, rules):
    """
    Finds the number of bindings to a training set on
    a given number of rules.

    Key arguments:
    cases -- the training set (cases) to look at.
    rules -- the rules analyze.
    """
    count = 0
    for case_id in cases:

        case = meta['data'][case_id]
        rules_success = True

        for rule in rules:

            rule_success = True

            for attribute, val in rule.iteritems():
                if attribute in case:

                    attr_success = False

                    if val[0] == ">=":
                        attr_success = val[1] <= case[attribute]
                    elif val[0] == "<=":
                        attr_success = val[1] >= case[attribute]

                    if not attr_success:
                        rule_success = False
                else:
                    rule_success = False

            if not rule_success:
                rules_success = False

        if rules_success:
            count += 1

    return count


def create_attrs(f):
    """
    Parses an attribute file into an attribute dictionary.

    Each key in the dictionary (attribute) contains a tuple
    that has the attribute's index in the file (line number)
    and its type or False if it is a value that is being
    predicted.

    e.g.:

        attrs['alcohol'] = (10, float)

    Key arguments:
    f -- the attribute file handle.
    """
    attrs = {}

    i = 0
    for line in f:
        # The first item is the attribute name.
        split = line[:-1].split(" ")
        attr = split[0]
        values = split[1:]
        # Continuous case.
        if len(values) == 1 and values[0].lower() in ['int', 'float', 'long', 'complex']:
            attrs[attr] = (i, values[0].lower())
        # Defining attribute case.
        else:
            v = {}
            for value in values:
                v[value] = 0
            attrs[attr] = (i, False)

        i += 1

    return attrs


def create_classes_and_data(f):

    # Data table.
    data = []

    # Classes dictionary.
    classes = {}

    # Build a reverse lookup dictionary for attribute index values.
    indices = {}
    for attr, value in meta['attrs'].iteritems():
        indices[value[0]] = attr

    # Case id counter.
    j = 0

    for line in f:
        case = {}
        # Index counter.
        i = 0
        split = line[:-1].split(" ")
        for value in split:
            # Find the attr name.
            attr = indices[i]
            # Check if this is the defining attribute, skip if so.
            if attr != meta['opts']['c']:

                value = eval(meta['attrs'][attr][1])(value)

                if attr in case:
                    case[attr].append(value)
                else:
                    case[attr] = value

            else:
                c = value

            i += 1

        # Create a new class if necessary.
        if c not in classes:
            classes[c] = []

        # Append the case to our test set.
        data.append(case)

        # And its id to class.
        classes[c].append(j)

        j += 1

    # Return classes dictionary and data table.
    return classes, data


def help():
    print("\n" +
          "The following are arguments required:\n" +
          "-e: the execution method (learn|classify)\n" +
          "-a: the attribute file location.\n" +
          "-c: the defining attribute in the attribute file.\n" +
          "-t: the training/testing file location.\n" +
          "-m: the model file (machine readable results).\n" +
          "-o: the output file (human readable results).\n" +
          "\n" +
          "Example Usage:\n" +
          "python irep.py -e learn -a \"../data/wine-attr.txt\"" +
          " -c color -t \"../data/wine-train.txt\" -m \"model.dat\"" +
          " -o \"wine-results.txt\"" +
          "\n" +
          "python irep.py -e classify -a \"../data/wine-attr.txt\"" +
          " -c color -t \"../data/wine-test.txt\" -m \"model.dat\"" +
          " -o \"wine-results.txt\"" +
          "\n")


if __name__ == "__main__":
    main()
