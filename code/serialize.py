# Serialize the data
from class_structure import Section, Expression, Statement, Information, Definition, Rule, Exemption, Reference

def serialize_section(section: Section) -> dict:
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
    return {
        "text": expr.text.lower(),
        "includes": [serialize_expression(e) for e in expr.includes],
        "sectionNumber": expr.section.sectionNumber if expr.section else None
    }

def serialize_reference(ref: Reference) -> dict:
    return {
        "text": ref.text.lower(),
        "target": get_statement_or_expression_id(ref.target),
        "sectionNumber": ref.section.sectionNumber if ref.section else None,
        "relationship": ref.relationship
    }

def serialize_statement(stmt: Statement) -> dict:
    # Because Statements have various subclasses (Definition, Information, Rule, Exemption),
    # you can detect the subclass and serialize accordingly:
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
    A helper that returns a short string ID for each Statement/Expression object
    if you want to keep track of references more precisely.
    For now, we can just return something like repr(obj) or id(obj).
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