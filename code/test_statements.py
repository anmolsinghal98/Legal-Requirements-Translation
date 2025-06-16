# This module provides functions to run code strings, serialize statements and references, and compare attribute values.

# # Import necessary classes from the class_structure module
from class_structure import Section, Expression, Statement, Information, Definition, Rule, Exemption, Reference
from serialize import serialize_statement, serialize_reference
from typing import List

def run_code_string(code_str: str) -> List[Section]:
    """
    Executes the given code string in a fresh namespace and
    returns a list of all Section objects created by that code.
    This function is designed to run code that defines sections, expressions, statements,
    information, definitions, rules, and exemptions.
    Args:
        code_str (str): The code string to execute.
    Returns:
        List[Section]: A list containing all Section objects created by the code.
    Raises:
        SyntaxError: If the code string has syntax errors.
        Exception: If any other error occurs during execution.
    """
    try:
        namespace = {}
        # Execute the code in an isolated namespace
        exec(code_str, globals(), namespace)

        information = [
            value
            for value in namespace.values()
            if isinstance(value, Information)
        ]
        definitions = [
                value
                for value in namespace.values()
                if isinstance(value, Definition)
            ]
        rules = [
                value
                for value in namespace.values()
                if isinstance(value, Rule)
            ]
        exemptions = [
                value
                for value in namespace.values()
                if isinstance(value, Exemption)
            ]
        return [information, definitions, rules, exemptions]
    except (SyntaxError, Exception) as e:
        print(e)
        return [[], [], [], []]
    
def run_code_string_references(code_str: str) -> List[Section]:
    """
    Executes the given code string in a fresh namespace and
    returns a list of all Reference objects created by that code.
    Args:
        code_str (str): The code string to execute.
    Returns:
        List[Section]: A list containing all Reference objects created by the code.
    Raises:
        SyntaxError: If the code string has syntax errors.
        Exception: If any other error occurs during execution.
    """
    try:
        namespace = {}
        # Execute the code in an isolated namespace
        exec(code_str, globals(), namespace)

        references = [
            value
            for value in namespace.values()
            if isinstance(value, Reference)
        ]

        statements = [
            value
            for value in namespace.values()
            if isinstance(value, Statement)
        ]
        return [references, statements]
    except (SyntaxError, Exception) as e:
        print(e)
        return [[], []]
    
def compare_strings_with_threshold(s1, s2, threshold=10):
    """
    Compare two strings using the edit distance algorithm.
    The edit distance is the minimum number of operations (insertions, deletions, substitutions)
    required to change one string into the other.
    Parameters:
      s1 (str): First string to compare.
      s2 (str): Second string to compare.
      threshold (int): Edit distance threshold for string comparisons.
    Returns:
      tuple: A tuple containing:
        - bool: True if the strings are considered similar (edit distance <= threshold), False otherwise.
        - int: The actual edit distance between the two strings.
    """
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    
    edit_distance = dp[m][n]
    return (edit_distance <= threshold, edit_distance)

def compare_list_strings_with_threshold(list1, list2, threshold=10):
    """
    Compare two lists of strings using the edit distance.
    
    Parameters:
      list1 (list): First list of strings.
      list2 (list): Second list of strings.
      threshold (int): Edit distance threshold for string comparisons.
      
    Returns:
      bool: True if every corresponding pair of strings are considered equal; False otherwise.
    """
    if len(list1) != len(list2):
        return False
    
    for s1, s2 in zip(list1, list2):
        similar, _ = compare_strings_with_threshold(s1, s2, threshold)
        if not similar:
            return False
    return True


def compare_serialized_expr(expr1, expr2, threshold=10):
    """
    Recursively compare two serialized Expression dictionaries.
    
    The dictionaries are assumed to have at least the following keys:
      - "text": a string (already lower-cased)
      - "includes": a list of nested serialized expressions or statements
      - "sectionNumber": an integer (or None)
    
    For string values, the comparison is based on the edit distance.
    For lists, elements are compared in order recursively.
    Other values are compared using standard equality.
    
    Parameters:
      expr1 (dict): First serialized expression.
      expr2 (dict): Second serialized expression.
      threshold (int): Edit distance threshold for string comparisons.
      
    Returns:
      bool: True if expr1 and expr2 are considered equal under these rules; False otherwise.
    """
    # Check that both dictionaries have the same keys.
    if set(expr1.keys()) != set(expr2.keys()):
        return False

    # Compare "text" using edit distance.
    if "text" in expr1 and "text" in expr2:
        if isinstance(expr1["text"], str) and isinstance(expr2["text"], str):
            similar, _ = compare_strings_with_threshold(expr1["text"], expr2["text"], threshold)
            if not similar:
                return False
        else:
            if expr1["text"] != expr2["text"]:
                return False

    # Compare "sectionNumber" using edit distance.
    if "sectionNumber" in expr1 and "sectionNumber" in expr2:
            similar, _ = compare_strings_with_threshold(str(expr1["sectionNumber"]), str(expr2["sectionNumber"]), threshold)
            if not similar:
                return False

    # Compare "includes" recursively.
    if "includes" in expr1 and "includes" in expr2:
        list1 = expr1["includes"]
        list2 = expr2["includes"]
        if len(list1) != len(list2):
            return False
        for item1, item2 in zip(list1, list2):
            # If the items are dictionaries, assume they are serialized expressions/statements.
            if isinstance(item1, dict) and isinstance(item2, dict):
                if not compare_serialized_expr(item1, item2, threshold):
                    return False
            # Otherwise, if they are strings, use the string comparison.
            elif isinstance(item1, str) and isinstance(item2, str):
                similar, _ = compare_strings_with_threshold(item1, item2, threshold)
                if not similar:
                    return False
            else:
                # For any other types, use direct equality.
                if item1 != item2:
                    return False

    return True

def compare_serialized_references(ref1, ref2, threshold=10):
    """
    Recursively compare two serialized Reference dictionaries.
    
    The dictionaries are assumed to have at least the following keys:
      - "text": a string (already lower-cased)
      - "target": a string (already lower-cased)
      - "sectionNumber": an integer (or None)
      - "relationship": a string

    For string values, the comparison is based on the edit distance.
    For lists, elements are compared in order recursively.
    Other values are compared using standard equality.

    Parameters:
        ref1 (dict): First serialized reference.
        ref2 (dict): Second serialized reference.

    Returns:
        bool: True if ref1 and ref2 are considered equal under these rules; False otherwise.
    """

    # Check that both dictionaries have the same keys.
    if set(ref1.keys()) != set(ref2.keys()):
        return False

    # Compare "text" using edit distance.
    if "text" in ref1 and "text" in ref2:
        if isinstance(ref1["text"], str) and isinstance(ref2["text"], str):
            similar, _ = compare_strings_with_threshold(ref1["text"], ref2["text"], threshold)
            if not similar:
                return False
        else:
            if ref1["text"] != ref2["text"]:
                return False

    # Compare "target" using edit distance.
    if "target" in ref1 and "target" in ref2:
        if isinstance(ref1["target"], str) and isinstance(ref2["target"], str):
            similar, _ = compare_strings_with_threshold(ref1["target"], ref2["target"], threshold)
            if not similar:
                return False
        else:
            if ref1["target"] != ref2["target"]:
                return False

    # Compare "sectionNumber" using edit distance.
    if "sectionNumber" in ref1 and "sectionNumber" in ref2:
        similar, _ = compare_strings_with_threshold(str(ref1["sectionNumber"]), str(ref2["sectionNumber"]), threshold)
        if not similar:
            return False

    # Compare "relationship" using edit distance.
    if "relationship" in ref1 and "relationship" in ref2:
        if isinstance(ref1["relationship"], str) and isinstance(ref2["relationship"], str):
            similar, _ = compare_strings_with_threshold(ref1["relationship"], ref2["relationship"], threshold)
            if not similar:
                return False
        else:
            if ref1["relationship"] != ref2["relationship"]:
                return False

    return True


def compare_serialized_expr_lists(list1, list2, threshold=10):
    """
    Compare two lists of serialized Expression dictionaries.
    
    Each list element is compared using `compare_serialized_expr` assuming
    that the ordering of expressions is consistent across the lists.
    
    Parameters:
      list1 (list): First list of serialized expressions.
      list2 (list): Second list of serialized expressions.
      threshold (int): Edit distance threshold for string comparisons.
      
    Returns:
      bool: True if every corresponding pair of dictionaries are considered equal; False otherwise.
    """
    if len(list1) != len(list2):
        return False
    
    for expr1, expr2 in zip(list1, list2):
        if not compare_serialized_expr(expr1, expr2, threshold):
            return False
    return True

def compare_serialized_references_lists(list1, list2, threshold=10):
    """
    Compare two lists of serialized Reference dictionaries.
    
    Each list element is compared using `compare_serialized_references` assuming
    that the ordering of references is consistent across the lists.
    
    Parameters:
        list1 (list): First list of serialized references.
        list2 (list): Second list of serialized references.

    Returns:
        bool: True if every corresponding pair of dictionaries are considered equal; False otherwise.
    """
    if len(list1) != len(list2):
        return False

    for ref1, ref2 in zip(list1, list2):
        if not compare_serialized_references(ref1, ref2, threshold):
            return False
    return True


def test_information_description(self, gt_code, generated_code):
    """
    Test case to compare the information description of two code snippets.
    Args:
        gt_code (str): The ground truth code snippet.
        generated_code (str): The generated code snippet to compare against the ground truth.
    Returns:
        tuple: A tuple containing two integers:
            - First integer: 1 if the test passed, 0 if it failed.
            - Second integer: 1 if the generated code has a true positive (TP), 0 if it has a true negative (TN), or 2 if it has no information description.
    """
    try:
        count1=0
        count2=0
        # 1. Run both snippets to get top-level sections
        list_s1 = run_code_string(gt_code)[0]
        list_s2 = run_code_string(generated_code)[0]
        # 2. Serialize them
        dict1 = [serialize_statement(s) for s in list_s1]
        dict2 = [serialize_statement(s) for s in list_s2]
        for i in range(len(dict1)):
            if dict1[i]['description'] != None:
                count1+=1
        for i in range(len(dict2)):
            if dict2[i]['description'] != None:
                count2+=1
        # compare length of dictionaries
        assert len(dict1) == len(dict2), "The two code snippets did not produce the same number of information statements."
        for i in range(len(dict1)):
            # 3. Compare
            assert compare_serialized_expr_lists(dict1[i]['description'], dict2[i]['description'], threshold=10), "The two code snippets did not produce the same information description."
        print("Info description test case passed")
        if count1 == 0: # if there are no descriptions in the ground truth code
            return 1, 0 # second value
        return 1, 1 # second value denotes TP
    except Exception as e:
        print(e)
        print("Info description test case failed")
        if count1 > count2:
            return 0,1
        elif count2 > count1:
            return 0,0
        else:
            return 0,2
    
def test_definition_term(self, gt_code, generated_code):
    """
    Test case to compare the defined terms of two code snippets.
    Args:
        gt_code (str): The ground truth code snippet.
        generated_code (str): The generated code snippet to compare against the ground truth.
    Returns:
        tuple: A tuple containing two integers:
            - First integer: 1 if the test passed, 0 if it failed.
            - Second integer: 1 if the generated code has a true positive (TP), 0 if it has a true negative (TN), or 2 if it has no defined terms.
    """
    try:
        count1 = 0
        count2 = 0
        # 1. Run both snippets to get top-level sections
        list_s1 = run_code_string(gt_code)[1]
        list_s2 = run_code_string(generated_code)[1]

        # 2. Serialize them
        dict1 = [serialize_statement(s) for s in list_s1]
        dict2 = [serialize_statement(s) for s in list_s2]

        for i in range(len(dict1)):
            if dict1[i]['defined_term'] is not None:
                count1 += 1
        for i in range(len(dict2)):
            if dict2[i]['defined_term'] is not None:
                count2 += 1
        # Compare length of top-level "definition statements"
        assert len(dict1) == len(dict2), "The two code snippets did not produce the same number of definition statements."

        # 3. Check that either the ground truth is a substring of the generated term OR vice versa
        for i in range(len(dict1)):

            assert dict1[i]['defined_term'] is not None, "Ground truth defined_term is None."
            assert dict2[i]['defined_term'] is not None, "Generated defined_term is None."

            assert compare_serialized_expr(dict1[i]['defined_term'], dict2[i]['defined_term'], threshold=10), "The two code snippets did not produce the same definition terms."

        print("Definition test case passed")

        # If there are no defined terms in the ground truth code
        if count1 == 0:
            return 1, 0  # Passed, True Negative

        return 1, 1  # Passed, True Positive

    except Exception as e:
        print(e)
        print("Definition test case failed")
        if count1 > count2:
            # Failed, but ground truth had definitions
            return 0, 1
        # Failed, ground truth had no definitions
        elif count2 > count1:
            return 0, 0
        return 0, 2
    

def test_definition_meaning(self, gt_code, generated_code):
    """ Test case to compare the meanings of defined terms in two code snippets.
    Args:
        gt_code (str): The ground truth code snippet.
        generated_code (str): The generated code snippet to compare against the ground truth.
    Returns:
        tuple: A tuple containing two integers:
            - First integer: 1 if the test passed, 0 if it failed.
            - Second integer: 1 if the generated code has a true positive (TP), 0 if it has a true negative (TN), or 2 if it has no meanings.
    """
    try:
        count1 = 0
        count2 = 0
        # 1. Run both snippets to get top-level sections
        list_s1 = run_code_string(gt_code)[1]
        list_s2 = run_code_string(generated_code)[1]

        # 2. Serialize them
        dict1 = [serialize_statement(s) for s in list_s1]
        dict2 = [serialize_statement(s) for s in list_s2]
        # Count how many "meaning" fields in the ground truth are non-empty
        for i in range(len(dict1)):
            if dict1[i]['meaning'] is not None and len(dict1[i]['meaning']) != 0:
                count1 += 1
        for i in range(len(dict2)):
            if dict2[i]['meaning'] is not None and len(dict2[i]['meaning']) != 0:
                count2 += 1

        # Compare the number of definition statements
        assert len(dict1) == len(dict2), "The two code snippets did not produce the same number of definition statements."
        
        for i in range(len(dict1)):
            assert dict1[i]['meaning'] is not None, "Ground truth meaning is None."
            assert dict2[i]['meaning'] is not None, "Generated meaning is None."

            ground_truth_meaning = dict1[i]['meaning']
            generated_meaning = dict2[i]['meaning']
            assert compare_serialized_expr_lists(ground_truth_meaning, generated_meaning, threshold=10), "The two code snippets did not produce the same definition meanings."

        print("Def meaning test case passed")

        # If there are no meanings in the ground truth code
        if count1 == 0:
            return 1, 0  # (Test passed, True Negative)

        return 1, 1      # (Test passed, True Positive)

    except Exception as e:
        print(e)
        print("Def meaning test case failed")

        # If an exception occurs and the ground truth code *did* have a meaning
        if count1 > count2:
            return 0, 1  # (Test failed, but ground truth had meaning)
        elif count2 > count1:
            return 0, 0 # (Test failed, ground truth had no meaning)
        return 0, 2      
    
def test_definition_exclusions(self, gt_code, generated_code):
    """ Test case to compare the exclusions of defined terms in two code snippets.
    Args:
        gt_code (str): The ground truth code snippet.
        generated_code (str): The generated code snippet to compare against the ground truth.
    Returns:
        tuple: A tuple containing two integers:
            - First integer: 1 if the test passed, 0 if it failed.
            - Second integer: 1 if the generated code has a true positive (TP), 0 if it has a true negative (TN), or 2 if it has no exclusions.
    """
    try:
        count1=0
        count2=0
        # 1. Run both snippets to get top-level sections
        list_s1 = run_code_string(gt_code)[1]
        list_s2 = run_code_string(generated_code)[1]
        # 2. Serialize them
        dict1 = [serialize_statement(s) for s in list_s1]
        dict2 = [serialize_statement(s) for s in list_s2]

        # compare length of dictionaries
        for i in range(len(dict1)):
            if len(dict1[i]['exclusions']) != 0:
                count1+=1
        for i in range(len(dict2)):
            if len(dict2[i]['exclusions']) != 0:
                count2+=1
        assert len(dict1) == len(dict2), "The two code snippets did not produce the same number of definition statements."
        for i in range(len(dict1)):
            assert compare_serialized_expr_lists(dict1[i]['exclusions'], dict2[i]['exclusions'], threshold=10), "The two code snippets did not produce the same definition exclusions."
        print("Def exclusions test case passed")
        if count1 == 0: # if there are no exclusions in the ground truth code
            return 1, 0 # second value denotes TN
        return 1, 1 # second value denotes
    except Exception as e:
        print(e)
        print("Def exclusions test case failed")
        if count1 > count2:
            return 0,1
        elif count2 > count1:
            return 0, 0
        return 0, 2
    
def test_rule_entity(self, gt_code, generated_code):
    """ Test case to compare the rule entities of two code snippets.
    Args:
        gt_code (str): The ground truth code snippet.
        generated_code (str): The generated code snippet to compare against the ground truth.
    Returns:
        tuple: A tuple containing two integers:
            - First integer: 1 if the test passed, 0 if it failed.
            - Second integer: 1 if the generated code has a true positive (TP), 0 if it has a true negative (TN), or 2 if it has no rule entities.
    """
    try:
        count1=0
        count2=0
        # 1. Run both snippets to get top-level sections
        list_s1 = run_code_string(gt_code)[2]
        list_s2 = run_code_string(generated_code)[2]
        # 2. Serialize them
        dict1 = [serialize_statement(s) for s in list_s1]
        dict2 = [serialize_statement(s) for s in list_s2]
        for i in range(len(dict1)):
            if dict1[i]['entity'] != None:
                count1+=1
        for i in range(len(dict2)):
            if dict2[i]['entity'] != None:
                count2+=1
        # compare length of dictionaries
        assert len(dict1) == len(dict2), "The two code snippets did not produce the same number of rule statements."
        for i in range(len(dict1)):
            # 3. Compare
            assert compare_serialized_expr(dict1[i]['entity'], dict2[i]['entity'], threshold=10), "The two code snippets did not produce the same rule entity."
        print("Rule entity test case passed")
        if count1 == 0: # if there are no entities in the ground truth code
            return 1, 0 # second value denotes TN
        return 1, 1 # second value denotes TP
    except Exception as e:
        print(e)
        print("Rule entity test case failed")
        if count1 > count2:
            return 0,1
        elif count2 > count1:
            return 0, 0
        return 0, 2

def test_rule_type(self, gt_code, generated_code):
    """ Test case to compare the rule types of two code snippets.
    Args:
        gt_code (str): The ground truth code snippet.
        generated_code (str): The generated code snippet to compare against the ground truth.
    Returns:
        tuple: A tuple containing two integers:
            - First integer: 1 if the test passed, 0 if it failed.
            - Second integer: 1 if the generated code has a true positive (TP), 0 if it has a true negative (TN), or 2 if it has no rule types.
    """
    try:
        count1=0
        count2=0
        # 1. Run both snippets to get top-level sections
        list_s1 = run_code_string(gt_code)[2]
        list_s2 = run_code_string(generated_code)[2]

        # 2. Serialize them
        dict1 = [serialize_statement(s) for s in list_s1]
        dict2 = [serialize_statement(s) for s in list_s2]
        for i in range(len(dict1)):
            if dict1[i]['rule_type'] != None:
                count1+=1
        for i in range(len(dict2)):
            if dict2[i]['rule_type'] != None:
                count2+=1
        # compare length of dictionaries
        assert len(dict1) == len(dict2), "The two code snippets did not produce the same number of rule statements."
        for i in range(len(dict1)):
            # 3. Compare
            assert dict1[i]['rule_type'] == dict2[i]['rule_type'], "The two code snippets did not produce the same rule type."
        print("Rule type test case passed")
        if count1 == 0: # if there are no rule types in the ground truth code
            return 1, 0 # second value denotes TN
        return 1, 1 # second value denotes TP
    except Exception as e:
        print(e)
        print("Rule type test case failed")
        if count1 > count2:
            return 0,1
        elif count2 > count1:
            return 0, 0
        return 0, 2
    
def test_rule_description(self, gt_code, generated_code):
    """ Test case to compare the rule descriptions of two code snippets.
    Args:
        gt_code (str): The ground truth code snippet.
        generated_code (str): The generated code snippet to compare against the ground truth.
    Returns:
        tuple: A tuple containing two integers:
            - First integer: 1 if the test passed, 0 if it failed.
            - Second integer: 1 if the generated code has a true positive (TP), 0 if it has a true negative (TN), or 2 if it has no rule descriptions.
    """
    try:
        count1=0
        count2=0
        # 1. Run both snippets to get top-level sections
        list_s1 = run_code_string(gt_code)[2]
        list_s2 = run_code_string(generated_code)[2]

        # 2. Serialize them
        dict1 = [serialize_statement(s) for s in list_s1]
        dict2 = [serialize_statement(s) for s in list_s2]
        for i in range(len(dict1)):
            if dict1[i]['description'] != None:
                count1+=1
        for i in range(len(dict2)):
            if dict2[i]['description'] != None:
                count2+=1
        # compare length of dictionaries
        assert len(dict1) == len(dict2), "The two code snippets did not produce the same number of rule statements."
        for i in range(len(dict1)):
            # 3. Compare
            assert compare_serialized_expr(dict1[i]['description'], dict2[i]['description'], threshold=10), "The two code snippets did not produce the same rule description."
        print("Rule description test case passed")
        if count1 == 0: # if there are no descriptions in the ground truth code
            return 1, 0 # second value denotes TN
        return 1, 1 # second value denotes TP
    except Exception as e:
        print(e)
        print("Rule description test case failed")
        if count1 > count2:
            return 0,1
        elif count2 > count1:
            return 0, 0
        return 0, 2

def test_rule_conditions(self, gt_code, generated_code):
    """ Test case to compare the rule conditions of two code snippets.
    Args:
        gt_code (str): The ground truth code snippet.
        generated_code (str): The generated code snippet to compare against the ground truth.
    Returns:
        tuple: A tuple containing two integers:
            - First integer: 1 if the test passed, 0 if it failed.
            - Second integer: 1 if the generated code has a true positive (TP), 0 if it has a true negative (TN), or 2 if it has no rule conditions.
    """
    try:
        count1=0
        count2=0
        # 1. Run both snippets to get top-level sections
        list_s1 = run_code_string(gt_code)[2]
        list_s2 = run_code_string(generated_code)[2]

        # 2. Serialize them
        dict1 = [serialize_statement(s) for s in list_s1]
        dict2 = [serialize_statement(s) for s in list_s2]
        for i in range(len(dict1)):
            if len(dict1[i]['conditions']) != 0:
                count1+=1
        for i in range(len(dict2)):
            if len(dict2[i]['conditions']) != 0:
                count2+=1
        # compare length of dictionaries
        assert len(dict1) == len(dict2), "The two code snippets did not produce the same number of rule statements."
        for i in range(len(dict1)):
            # 3. Compare
            assert compare_serialized_expr_lists(dict1[i]['conditions'], dict2[i]['conditions'], threshold=10), "The two code snippets did not produce the same rule conditions."
        print("Rule condition test case passed")
        if count1 == 0: # if there are no conditions in the ground truth code
            return 1, 0 # second value denotes TN
        else: return 1, 1 # second value denotes TP
    except Exception as e:
        print(e)
        print("Rule condition test case failed")
        if count1 > count2:
            return 0,1
        elif count2 > count1:
            return 0, 0
        return 0, 2

def test_exemption_description(self, gt_code, generated_code):
    """ Test case to compare the exemption descriptions of two code snippets.
    Args:
        gt_code (str): The ground truth code snippet.
        generated_code (str): The generated code snippet to compare against the ground truth.
    Returns:
        tuple: A tuple containing two integers:
            - First integer: 1 if the test passed, 0 if it failed.
            - Second integer: 1 if the generated code has a true positive (TP), 0 if it has a true negative (TN), or 2 if it has no exemption descriptions.
    """
    try:
        count1=0
        count2=0
        # 1. Run both snippets to get top-level sections
        list_s1 = run_code_string(gt_code)[3]
        list_s2 = run_code_string(generated_code)[3]
        # 2. Serialize them
        dict1 = [serialize_statement(s) for s in list_s1]
        dict2 = [serialize_statement(s) for s in list_s2]
        # compare length of dictionaries
        for i in range(len(dict1)):
            if len(dict1[i]['description']) != 0:
                count1+=1
        for i in range(len(dict2)):
            if len(dict2[i]['description']) != 0:
                count2+=1
        assert len(dict1) == len(dict2), "The two code snippets did not produce the same number of exemption statements."
        for i in range(len(dict1)):
            # 3. Compare
            assert compare_serialized_expr_lists(dict1[i]['description'], dict2[i]['description'], threshold=10), "The two code snippets did not produce the same exemption description."
        print("Exemption description test case passed")
        if count1 == 0: # if there are no exemptions in the ground truth code
            return 1, 0 # second value denotes TN
        else: return 1, 1 # second value denotes TP
        return 1
    except Exception as e:
        print(e)
        print("Exemption test case failed")
        if count1 > count2:
            return 0,1
        elif count2 > count1:
            return 0, 0
        return 0, 2
    
def test_reference_relationship(self, gt_code, generated_code):
    """ Test case to compare the reference relationships of two code snippets.
    Args:
        gt_code (str): The ground truth code snippet.
        generated_code (str): The generated code snippet to compare against the ground truth.
    Returns:
        tuple: A tuple containing two integers:
            - First integer: 1 if the test passed, 0 if it failed.
            - Second integer: 1 if the generated code has a true positive (TP), 0 if it has a true negative (TN), or 2 if it has no references.
    """
    try:
        count1=0
        count2=0
        # 1. Run both snippets to get top-level sections
        list_s1 = run_code_string_references(gt_code)[0]
        list_s2 = run_code_string_references(generated_code)[0]

        # 2. Serialize them
        dict1 = [serialize_reference(s) for s in list_s1]
        dict2 = [serialize_reference(s) for s in list_s2]
        for i in range(len(dict1)):
            if len(dict1) != 0:
                count1 += 1
        for i in range(len(dict2)):
            if len(dict2) != 0:
                count2 += 1
        # compare length of dictionaries
        assert len(dict1) == len(dict2), "The two code snippets did not produce the same number of references."
        for i in range(len(dict1)):
            # 3. Compare

            assert compare_serialized_references(dict1[i], dict2[i], threshold=10), "The two code snippets did not produce the same references."
        print(f"references test case passed")
        if count1==0: # no refines assigned in gt
            return 1, 0 # second value denotes TN
        return 1, 1 # second value denotes TP
    except Exception as e:
        print(e)
        print(f"references test case failed")
        if count1 > count2:
            return 0,1
        elif count2 > count1:
            return 0, 0
        return 0, 2

def test_statement_relationship(self, gt_code, generated_code, relation):
    """ Test case to compare the statement relationships of two code snippets.
    Args:
        gt_code (str): The ground truth code snippet.
        generated_code (str): The generated code snippet to compare against the ground truth.
        relation (str): The relationship to compare (e.g., "refines", "is_a", etc.).
    Returns:
        tuple: A tuple containing two integers:
            - First integer: 1 if the test passed, 0 if it failed.
            - Second integer: 1 if the generated code has a true positive (TP), 0 if it has a true negative (TN), or 2 if it has no relationships.
    """
    try:
        count1=0
        count2=0
        # 1. Run both snippets to get top-level sections
        list_s1 = run_code_string_references(gt_code)[1]
        list_s2 = run_code_string_references(generated_code)[1]

        # 2. Serialize them
        dict1 = [serialize_statement(s) for s in list_s1]
        dict2 = [serialize_statement(s) for s in list_s2]
        for i in range(len(dict1)):
            if len(dict1[i]['relationships'][relation]) != 0:
                count1 += 1
        for i in range(len(dict2)):
            if len(dict2[i]['relationships'][relation]) != 0:
                count2 += 1
        # compare length of dictionaries
        assert len(dict1) == len(dict2), f"The two code snippets did not produce the same number of statements."
        for i in range(len(dict1)):
            # 3. Compare

            assert compare_list_strings_with_threshold(dict1[i]['relationships'][relation], dict2[i]['relationships'][relation], threshold=10), f"The two code snippets did not produce the same {relation} relationship list for each statement."
        print(f"{relation} test case passed")
        if count1==0: # no refines assigned in gt
            return 1, 0 # second value denotes TN
        return 1, 1 # second value denotes TP
    except Exception as e:
        print(e)
        print(f"{relation} test case failed")
        if count1 > count2:
            return 0,1
        elif count2 > count1:
            return 0, 0
        return 0, 2
        