# This module provides functions to serialize various classes from the class_structure module into a dictionary format.
# It includes serialization for Section, Expression, Statement, Information, Definition, Rule, Exemption, and Reference classes.

# Import necessary classes from the class_structure module
from class_structure import Section, Expression, Statement, Information, Definition, Rule, Exemption, Reference

def serialize_section(section: Section) -> dict:
    """
    Serialize a Section object into a dictionary format.
    Args:
        section (Section): The Section object to serialize.
    Returns:
        dict: A dictionary representation of the Section object.
    """
    return {
        "sectionNumber": section.sectionNumber,
        "sectionTitle": section.sectionTitle,
        "subSections": [
            serialize_section(sub) for sub in section.subSections
        ],
        "expressions": [
            serialize_expression(expr) for expr in section.expressions
        ],
        "statements": [
            serialize_statement(stmt) for stmt in section.statements
        ]
    }

def serialize_expression(expr: Expression) -> dict:
    """
    Serialize an Expression object into a dictionary format.
    Args:
        expr (Expression): The Expression object to serialize.
    Returns:
        dict: A dictionary representation of the Expression object.
    """
    return {
        "text": expr.text.lower(),
        "includes": [serialize_expression(e) for e in expr.includes],
        "sectionNumber": expr.section.sectionNumber if expr.section else None
    }

def serialize_reference(ref: Reference) -> dict:
    """
    Serialize a Reference object into a dictionary format.
    Args:
        ref (Reference): The Reference object to serialize.
    Returns:
        dict: A dictionary representation of the Reference object.
    """
    return {
        "text": ref.text.lower(),
        "target": get_statement_or_expression_id(ref.target),
        "sectionNumber": ref.section.sectionNumber if ref.section else None,
        "relationship": ref.relationship
    }

def serialize_statement(stmt: Statement) -> dict:
    """
    Serialize a Statement object into a dictionary format.
    This function handles different subclasses of Statement and extracts relevant fields.
    Args:
        stmt (Statement): The Statement object to serialize.
    Returns:
        dict: A dictionary representation of the Statement object.
    """
    base = {
        "section": stmt.sections.sectionNumber if stmt.sections else None,
        "relationships": {
            # relationships are references to other Statement/Expression objects,
            # so you may need to gather their identifiers or convert them fully
            "refines": [get_statement_or_expression_id(x) for x in stmt.relationships["refines"]],
            "is_refined_by": [get_statement_or_expression_id(x) for x in stmt.relationships["is_refined_by"]],
            "has_exception": [get_statement_or_expression_id(x) for x in stmt.relationships["has_exception"]],
            "is_exception_to": [get_statement_or_expression_id(x) for x in stmt.relationships["is_exception_to"]],
            "follows": [get_statement_or_expression_id(x) for x in stmt.relationships["follows"]],
            "is_followed_by": [get_statement_or_expression_id(x) for x in stmt.relationships["is_followed_by"]],
        }
    }
    
    if isinstance(stmt, Information):
        base["type"] = "Information"
        base["description"] = [serialize_expression(d) for d in stmt.description]
    elif isinstance(stmt, Definition):
        base["type"] = "Definition"
        base["defined_term"] = serialize_expression(stmt.defined_term)
        base["meaning"] = [serialize_expression(m) for m in stmt.meaning]
        base["exclusions"] = [serialize_expression(e) for e in stmt.exclusions]
    elif isinstance(stmt, Rule):
        base["type"] = "Rule"
        base["rule_type"] = stmt.rule_type
        base["entity"] = serialize_expression(stmt.entity)
        base["description"] = serialize_expression(stmt.description) if stmt.description else None
        base["conditions"] = [serialize_expression(c) for c in stmt.conditions]
    elif isinstance(stmt, Exemption):
        base["type"] = "Exemption"
        base["description"] = [serialize_expression(d) for d in stmt.description]
    else:
        base["type"] = "UnknownStatementSubclass"

    return base


def get_statement_or_expression_id(obj):
    """
    Get a unique identifier for a Statement or Expression object.
    Args:
        obj (Statement or Expression): The object to get the identifier for.
    Returns:
        str: A string representation of the object, typically its text or a unique identifier.
    """
    if isinstance(obj, Reference):
        return obj.text.lower()
    elif isinstance(obj, Rule):
        return obj.entity.text.lower()
    elif isinstance(obj, Information):
        return obj.description[0].text.lower()
    elif isinstance(obj, Definition):
        return obj.defined_term.text.lower()
    elif isinstance(obj, Exemption):
        return obj.description[0].text.lower()
    else:
        return repr(obj)