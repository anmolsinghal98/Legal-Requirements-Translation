from typing import List, Optional, Union

class Section:
    """
    A bullet point in the legal text. Every bullet point starts a new Section,
    and sub-bullet points become subSections.

    Attributes:
        sectionNumber (str): The identifying number or label of this Section.
        sectionTitle (str): An optional title for this Section.
        parent (Optional[Section]): The parent Section if this is a nested (sub-)Section,
            otherwise None for a top-level Section.
        subSections (List[Section]): Any child Sections nested under this Section.
        expressions (List[Expression]): The Expression objects contained directly in this Section.
        statements (List[Statement]): The Statement objects contained directly in this Section.

    Methods:
        add_subsection(subsection: 'Section'):
            Adds a subsection (child) to this Section and sets the subsection's parent to self.

        add_expression(expression: 'Expression'):
            Adds an Expression object to this Section’s expressions list.

        add_statement(statement: 'Statement'):
            Adds a Statement object to this Section’s statements list.
    """

    def __init__(self,sectionNumber: str, sectionTitle: str = "", parent=None):
        self.sectionNumber: str = sectionNumber
        self.sectionTitle: str = sectionTitle
        self.parent: Optional['Section'] = parent
        self.subSections: List['Section'] = []
        self.expressions: List['Expression'] = []
        self.statements: List['Statement'] = []

    def add_subsection(self, subsection: 'Section'):
        self.subSections.append(subsection)
        subsection.parent = self
    def add_expression(self, expression: 'Expression'):
        self.expressions.append(expression)
    def add_statement(self, statement: 'Statement'):
        self.statements.append(statement)


class Expression:
    """
    A snippet of text within one bullet point (Section). Represents the smallest
    textual unit that can contain references or other embedded elements.

    Each Expression belongs to exactly one Section.

    Attributes:
        section (Section): The Section in which this Expression is found.
        text (str): The textual content of the Expression.
        includes (Optional[List[Expression]]): A child Expression in a subsection that this Expression includes
    """

    def __init__(self, section: Section, text: str, includes=None):
        self.section: Section = section
        section.add_expression(self)
        self.text: str = text
        self.includes: Optional[List[Expression]] = includes if includes is not None else []


class Reference(Expression):
    """
    A type of Expression that refers to another part of the legal text.

    Attributes:
        target (Union[Expression, Statement]): The target Expression or Statement that this Reference points to.
    """

    def __init__(self, section: Section, text: str, target):
        super().__init__(section, text)
        self.target: Statement = target
        self.relationship: Optional[str] = None


class Statement:
    """
    A legal statement that can span multiple bullet points (Sections) if those
    bullet points are nested under a single conceptual clause. Statements often
    contain or refer to multiple Expressions.

    Attributes:
        section (Section): The Section that represents
            the location in the text where this Statement starts.
        relationships (dict of str -> List[Expression or Statement]): A dictionary of
            six possible relationship types, each mapping to a list of Expressions or Statements
            that are connected to this Statement or references that are present within the statement 
            in the specified manner.

    Relationship keys:
        - "refines": A list of References or Statements this Statement refines
            (providing more detail about).
        - "is_refined_by": A list of References or Statements that refine this Statement.
        - "has_exception": A list of Expressions, References, or Statements that are exceptions to this Statement.
        - "is_exception_to": A list of References or Statements for which this Statement is an exception.
        - "follows": A list of References or Statements that precede (act as post-conditions to) this Statement.
        - "is_followed_by": A list of References or Statements that follow this Statement.

    Methods:
        add_refines(target): Adds a target to the "refines" relationship.
        add_exception(exception): Adds a target to the "has_exception" relationship.
        add_follows(target): Adds a target to the "follows" relationship.
        add_is_refined_by(target): Adds a target to the "is_refined_by" relationship.
        add_is_exception_to(exception): Adds a target to the "is_exception_to" relationship.
        add_is_followed_by(target): Adds a target to the "is_followed_by" relationship.
    """

    def __init__(self, section: Optional[Section] = None):
        self.sections: Section = section
        self.relationships = {
            "refines": [],
            "is_refined_by": [],
            "has_exception": [],
            "is_exception_to": [],
            "follows": [],
            "is_followed_by": []
        }

    def add_refines(self, target: Union['Reference', 'Statement']):
        self.relationships["refines"].append(target)
        if isinstance(target, Reference):
            target.relationship = "refines"
    def add_exception(self, exception: Union['Expression', 'Statement']):
        self.relationships["has_exception"].append(exception)
        if isinstance(exception, Reference):
            exception.relationship = "has_exception"
    def add_follows(self, target: Union['Reference', 'Statement']):
        self.relationships["follows"].append(target)
        if isinstance(target, Reference):
            target.relationship = "follows"
    def add_is_refined_by(self, target: Union['Reference', 'Statement']):
        self.relationships["is_refined_by"].append(target)
        if isinstance(target, Reference):
            target.relationship = "is_refined_by"
    def add_is_exception_to(self, target: Union['Reference', 'Statement']):
        self.relationships["is_exception_to"].append(target)
        if isinstance(target, Reference):
            target.relationship = "is_exception_to"
    def add_is_followed_by(self, target: Union['Reference', 'Statement']):
        self.relationships["is_followed_by"].append(target)
        if isinstance(target, Reference):
            target.relationship = "is_followed_by"


class Information(Statement):
    """
    A type of Statement that represents something that is known or proved to be true.

    Attributes:
        description (List[Expression]): The Expressions that contains the factual information.
    """

    def __init__(self, section, description: Expression):
        super().__init__(section)
        self.description: List[Expression] = []
        if description is not None:
            self.description.append(description)


class Definition(Statement):
    """
    A type of Statement that defines a concept or term in the legal text.

    Attributes:
        defined_term (Expression): The Expression stating the term being defined.
        meaning (List[Expression]): One or more Expressions elaborating the meaning of the term.
        exclusions (List[Expression]): Expressions clarifying what the term excludes or does not cover.
    """

    def __init__(self, section, defined_term: Expression):
        super().__init__(section)
        self.defined_term: Expression = defined_term
        self.meaning: List[Expression] = []
        self.exclusions: List[Expression] = []


class Rule(Statement):
    """
    A Statement describing a legal rule, which may take one of four types: obligation,
    permission, prohibition, or penalty.

    Attributes:
        rule_type (int): An integer indicating which type of rule. Should be one of:
            OBLIGATION, PERMISSION, PROHIBITION, PENALTY.
        entity (Expression): The main entity (person, object, etc.) to which the rule applies.
        description (Expression): An Expression describing the rule.
        conditions (List[Expression]): Expressions indicating the conditions under which the rule applies.
    """

    OBLIGATION = 0
    PERMISSION = 1
    PROHIBITION = 2
    PENALTY = 3

    def __init__(self, section, entity: Expression):
        super().__init__(section)
        self.rule_type: int = None
        self.entity: Expression = entity
        self.description: Optional[Expression] = None
        self.conditions: List[Expression] = []


class Exemption(Statement):
    """
    A type of Statement indicating that a person, object, or situation is exempt
    from another rule or requirement.

    Attributes:
        description (List[Expression]): One or more Expressions describing the exemption.
    """

    def __init__(self, section=None, description: Optional[Expression] = None):
        super().__init__(section)
        self.description: List[Expression] = []
        if description is not None:
            self.description.append(description)