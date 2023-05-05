import copy
import operator
import re

from tree_sitter import Language, Parser
from pre_train.utils import args

GO = args.go
JAVA = args.java
JS = args.javascript
PHP = args.php
PY = args.python
RUBY = args.ruby

LANGUAGE = {
    GO: Language(args.go_ast_file, 'go'),
    JAVA: Language(args.java_ast_file, 'java'),
    JS: Language(args.javascript_ast_file, 'javascript'),
    PHP: Language(args.php_ast_file, 'php'),
    PY: Language(args.python_ast_file, 'python'),
    RUBY: Language(args.ruby_ast_file, 'ruby'),
}

parser = Parser()

SOURCE_PREFIX_POSTFIX = {
    JAVA: ['class A{ ', ' }'],

    PHP: ['<?php class A { ', ' } ?>'],
}

# java and php can be parsed with special beginning and end
PATTERNS_METHOD_ROOT = {
    JAVA: """
    (program
        (class_declaration
            body: (class_body
                (method_declaration) @method_root)
        )
    )
    """,

    PHP: """
    (program
        (class_declaration
            body: (declaration_list
                (method_declaration) @method_root)
        )
    )
    """
}

PATTERNS_METHOD_HEAD = {
    GO: """
    (source_file
        (function_declaration) @head
    )
    """,

    RUBY: """
    [
        (program
            (method
                name: (identifier) @head
            )
        )
        (program
            (method
                parameters: (method_parameters) @head
            )
        )
        (program
            (method
                name: (constant) @head
            )
        )
    ]
    """,

}

PATTERNS_METHOD_BODY = {
    GO: """
    (source_file
        [
        (function_declaration
            body: (block) @body)
        (method_declaration
            body: (block) @body)
        ]
    )
    """,

    JAVA: """
    (method_declaration
        body: (block) @body
    )
    """,

    JS: """
    [
        (program
            (function_declaration
                body: (statement_block) @body
            )
        )
        (program
            (expression_statement
                (function
                    body: (statement_block) @body
                )
            )
        )
        (program
            (expression_statement
                (assignment_expression
                    right: (function
                        body: (statement_block) @body
                    )
                )
            )
        )
    ]
    """,

    PHP: """
    (method_declaration
        body: (compound_statement) @body
    )
    """,

    PY: """
    (module
        (function_definition
            body: (block) @body
        )
    )
    """,
}

PATTERNS_METHOD_NAME = {
    GO: """
    [
        (source_file
            (method_declaration
                name: (field_identifier) @method_name
            )
        )
        (source_file
            (function_declaration
                name: (identifier) @method_name
            )
        )
    ]
    """,

    JAVA: """
    (method_declaration
        name: (identifier) @method_name
    )
    """,

    JS: """
    (program
        (function_declaration
            name: (identifier) @method_name
        )
    )
    """,

    PHP: """
    (function_definition
        name: (name) @method_name
    )
    """,

    PY: """
    (module
        (function_definition
            name: (identifier) @method_name
        )
    )
    """,

    RUBY: """
    (program
        (method
            name: (identifier) @method_name
        )
    )
    """,
}

PATTERNS_METHOD_INVOCATION = {
    GO: """
    [
        (call_expression
            function: (selector_expression
                field: (field_identifier) @method_invocation
            )
        )
        (call_expression
            function: (identifier) @method_invocation
        )
    ]
    """,

    JAVA: """
    (method_invocation
        name: (identifier) @method_invocation
    )
    """,

    JS: """
    [
        (call_expression
            function: (member_expression
                property: (property_identifier) @method_invocation
            )
        )
        (call_expression
            function: (identifier) @method_invocation
        )
    ]
    """,

    PHP: """
    [
        (scoped_call_expression
            name: (name) @method_invocation
        )
        (function_call_expression
            (name) @method_invocation
        )
        (member_call_expression
            name: (name) @method_invocation
        )
        (object_creation_expression
            (qualified_name
                (name) @method_invocation
            )
        )
    ]
    """,

    PY: """
    [
        (call
            function: (identifier) @method_invocation
        )
        (call
            function: (attribute
                attribute: (identifier) @method_invocation
            )
        )
    ]
    """,

    RUBY: """
    (call
        method: (identifier) @method_invocation
    )
    """,
}

PATTERNS_IDENTIFIER = {
    GO: """
    [
        (identifier) @identifier_name
        package: (package_identifier) @pkg_name
    ]
    """,

    JAVA: """
        (identifier) @identifier_name
    """,

    JS: """
    (identifier) @identifier_name
    """,

    PHP: """
    [
        (variable_name) @identifier_name
        (dynamic_variable_name) @dynamic_identifier_name
    ]
    """,

    PY: """
        (identifier) @identifier_name
    """,

    RUBY: """
    [
        (identifier) @identifier_name
        (instance_variable) @identifier_name
        (class_variable) @identifier_name
    ]
    """,
}

PATTERNS_ATTRIBUTE = {
    GO: """
    [
        (call_expression
            function: (selector_expression
                field: (field_identifier) @method_invocation
            )
        )
        (call_expression
            function: (identifier) @method_invocation
        )
    ]
    """,

    JAVA: """
    [
        (field_access
            field: (identifier) @attribute_name
        )
        (method_invocation
            name: (identifier) @method_invocation
        )
    ]
    """,

    JS: """
    [
        (call_expression
            function: (member_expression
                property: (property_identifier) @method_invocation
            )
        )
        (call_expression
            function: (identifier) @method_invocation
        )
    ]
    """,

    PHP: """
    [
        (scoped_call_expression
            name: (name) @method_invocation
        )
        (function_call_expression
            (name) @method_invocation
        )
        (member_call_expression
            name: (name) @method_invocation
        )
        (object_creation_expression
            (qualified_name
                (name) @method_invocation
            )
        )
    ]
    """,

    PY: """
    [    
        (attribute
            attribute: (identifier) @attribute_name
        )
        (call
            function: (identifier) @method_invocation
        )
        (keyword_argument
                name: (identifier) @attribute_name
        )
    ]
    """,

    RUBY: """
    (call
        method: (identifier) @method_invocation
    )
    """,
}

# include comment and string
PATTERNS_STRING = {
    GO: """
    [
        (comment) @comment
        (interpreted_string_literal) @string
    ]
    """,

    JAVA: """
    [
        (block_comment) @comment
        (line_comment) @comment
        (string_literal) @string
    ]
    """,

    JS: """
    [
        (comment) @comment
        (string) @string
    ]
    """,

    PHP: """
    [
        (comment) @comment
        (string) @string
        (encapsed_string) @string
    ]
    """,

    PY: """
    [
        (comment) @comment
        (string) @string
    ]
    """,

    RUBY: """
    [
        (comment) @comment
        (string) @string
    ]
    """,
}

PATTERNS_STRING_SPECIAL = {
    RUBY: """
    [
        (regex) @string
    ]
    """,
}


def parse_ast(source, lang):
    """
    Parse the given code into corresponding ast.
    Args:
        source (str): code in string
        lang (str): Set the language

    Returns:
        tree_sitter_tools.Node: Method/Function root node
    """
    parser.set_language(LANGUAGE[lang])
    if lang in SOURCE_PREFIX_POSTFIX:
        source = SOURCE_PREFIX_POSTFIX[lang][0] + source + SOURCE_PREFIX_POSTFIX[lang][1]
    tree = parser.parse(source.encode('utf-8'))
    root = tree.root_node

    if lang in PATTERNS_METHOD_ROOT:
        query = LANGUAGE[lang].query(PATTERNS_METHOD_ROOT[lang])
        captures = query.captures(root)
        try:
            root = captures[0][0]
        except:
            return None
    return root


def get_node_name(source, node, lang):
    """
    Get node name, for php is shifted by prefix.

    Args:
        source (str): Source code string
        node (tree_sitter_tools.Node): Node instance
        lang (str): Source code language

    Returns:
        str: Name of node

    """
    if node.is_named:
        if lang in SOURCE_PREFIX_POSTFIX:
            return source[node.start_byte - len(SOURCE_PREFIX_POSTFIX[lang][0]):
                          node.end_byte - len(SOURCE_PREFIX_POSTFIX[lang][0])]
        else:
            return source[node.start_byte: node.end_byte]
    return ''


def get_node_position(node, lang) -> tuple:
    """
    Get node position, for php is shifted by prefix.

    Args:
        node (tree_sitter_tools.Node): Node instance
        lang (str): Source code language

    Returns:
        int: Start position of node
        int: End position of node
    """
    if node.is_named:
        if lang in SOURCE_PREFIX_POSTFIX:
            return node.start_byte - len(SOURCE_PREFIX_POSTFIX[lang][0]), node.end_byte - len(
                SOURCE_PREFIX_POSTFIX[lang][0])
        else:
            return node.start_byte, node.end_byte
    return 0, 0


def get_head_position(root, lang):
    """
    Get the position of the definition of function, for php is shifted by prefix.

    Args:
        root (tree_sitter_tools.Node): Node instance
        lang (str): Source code language

    Returns:
        int: Start position of definition
        int: End position of definition
    """

    if lang in PATTERNS_METHOD_BODY:
        query = LANGUAGE[lang].query(PATTERNS_METHOD_BODY[lang])
        # print(root.sexp())
        captures = query.captures(root)
        if captures:
            node = captures[0][0]
            if node.is_named:
                if lang in SOURCE_PREFIX_POSTFIX:
                    return 0, node.start_byte - len(SOURCE_PREFIX_POSTFIX[lang][0])
                else:
                    return 0, node.start_byte

        elif lang in PATTERNS_METHOD_HEAD:
            query = LANGUAGE[lang].query(PATTERNS_METHOD_HEAD[lang])
            captures = query.captures(root)
            if captures:
                node = captures[0][0]
                if node.is_named:
                    if lang in SOURCE_PREFIX_POSTFIX:
                        return 0, node.end_byte - len(SOURCE_PREFIX_POSTFIX[lang][0])
                    else:
                        return 0, node.end_byte
                else:
                    return 0, 0
            else:
                return 0, 0
        else:
            return 0, 0

    # ruby has no body node
    if lang == 'ruby':
        query = LANGUAGE[lang].query(PATTERNS_METHOD_HEAD[lang])
        captures = query.captures(root)
        # ruby function with params
        # node = None
        if len(captures) == 2:
            node = captures[1][0]
        # ruby function without params
        elif len(captures) == 1:
            node = captures[0][0]
        else:
            return 0, 0
        return 0, node.end_byte

    # perhaps there are other langs or some unknown bugs thus, here we throw exceptions


def extract_variable_from_code(source, lang):
    """
    Args:
        source: :(str) Source code string
        lang: :(str) Source code language
    Returns:
        mask_var_str_code (str): Source code with variables replaced by a special token and strings by another
        mask_str_code (str): Source code with strings replaced by a special token
        code_variables (str): All the variables in a function
        code_strings (str): All the strings in a function
    """
    mask_var_str_code = copy.deepcopy(source)
    mask_str_code = copy.deepcopy(source)
    root = parse_ast(source=source, lang=lang)

    assert root is not None and "ERROR" not in root.sexp(), f"code pre-processed error"

    # extract variables
    identifier_nodes = []
    if lang in PATTERNS_IDENTIFIER:
        query = LANGUAGE[lang].query(PATTERNS_IDENTIFIER[lang])
        captures = query.captures(root)
        identifier_nodes = [get_node_position(node=capture[0], lang=lang) for capture in captures]

    # deduplication
    identifier_nodes = list(set(identifier_nodes))

    # extract attributes
    attr_nodes = []
    if lang in PATTERNS_ATTRIBUTE:
        query = LANGUAGE[lang].query(PATTERNS_ATTRIBUTE[lang])
        captures = query.captures(root)
        attr_nodes = [get_node_position(node=capture[0], lang=lang) for capture in captures]

    # deduplication
    attr_nodes = list(set(attr_nodes))

    # extract strings
    string_nodes = []
    if lang in PATTERNS_STRING:
        query = LANGUAGE[lang].query(PATTERNS_STRING[lang])
        captures = query.captures(root)
        string_nodes = [get_node_position(node=capture[0], lang=lang) for capture in captures]
    string_nodes.sort(key=operator.itemgetter(0), reverse=True)

    # deduplication
    string_nodes = list(set(string_nodes))

    _, head_node_end_byte = get_head_position(root, lang)
    assert not head_node_end_byte == 0, f"parse code head fail"

    # remove error tokens

    identifier_nodes_copy = copy.deepcopy(identifier_nodes)

    for str_start_byte, str_end_byte in identifier_nodes_copy:
        for sub_str_start_byte, sub_str_end_byte in identifier_nodes_copy:
            if (sub_str_start_byte >= str_start_byte and sub_str_end_byte < str_end_byte) or (
                    sub_str_start_byte > str_start_byte and sub_str_end_byte <= str_end_byte):
                if (sub_str_start_byte, sub_str_end_byte) in identifier_nodes:
                    identifier_nodes.remove((sub_str_start_byte, sub_str_end_byte))

    attr_nodes_copy = copy.deepcopy(attr_nodes)

    for str_start_byte, str_end_byte in attr_nodes_copy:
        for sub_str_start_byte, sub_str_end_byte in attr_nodes_copy:
            if (sub_str_start_byte >= str_start_byte and sub_str_end_byte < str_end_byte) or (
                    sub_str_start_byte > str_start_byte and sub_str_end_byte <= str_end_byte):
                if (sub_str_start_byte, sub_str_end_byte) in attr_nodes:
                    attr_nodes.remove((sub_str_start_byte, sub_str_end_byte))

    useful_string_node = copy.deepcopy(string_nodes)

    for str_start_byte, str_end_byte in string_nodes:
        for sub_str_start_byte, sub_str_end_byte in string_nodes:
            if (sub_str_start_byte >= str_start_byte and sub_str_end_byte < str_end_byte) or (
                    sub_str_start_byte > str_start_byte and sub_str_end_byte <= str_end_byte):
                if (sub_str_start_byte, sub_str_end_byte) in useful_string_node:
                    useful_string_node.remove((sub_str_start_byte, sub_str_end_byte))

    # remove identifiers in attr_nodes
    for attr_start_byte, attr_end_byte in attr_nodes:
        if (attr_start_byte, attr_end_byte) in identifier_nodes:
            identifier_nodes.remove((attr_start_byte, attr_end_byte))

    # remove identifiers in strings
    # remove strings in variables
    identifier_nodes_copy = copy.deepcopy(identifier_nodes + attr_nodes)
    useful_string_node_copy = copy.deepcopy(useful_string_node)
    for str_start_byte, str_end_byte in useful_string_node_copy:
        for id_start_byte, id_end_byte in identifier_nodes_copy:
            if id_start_byte >= str_start_byte and id_end_byte <= str_end_byte:
                if (id_start_byte, id_end_byte) in identifier_nodes:
                    identifier_nodes.remove((id_start_byte, id_end_byte))

            # e.g. foreach (static::${'_'.$rel_name} as $key => $settings)   in php language
            if str_start_byte >= id_start_byte and str_end_byte <= id_end_byte:
                if (str_start_byte, str_end_byte) in useful_string_node:
                    useful_string_node.remove((str_start_byte, str_end_byte))

    removed_code_segments = identifier_nodes + useful_string_node
    removed_code_segments.sort(key=operator.itemgetter(0), reverse=True)

    code_variables = []
    code_strings = []

    for (var_start_byte, var_end_byte) in removed_code_segments:
        # keep the head of a function
        if var_start_byte >= head_node_end_byte:

            if (var_start_byte, var_end_byte) in identifier_nodes:
                # deal with the special string replaced token we defined in the project
                # pay respect to the idea from tz
                mask_var_str_code = mask_var_str_code[0: var_start_byte] + \
                                    args.var_replaced_token + \
                                    mask_var_str_code[var_end_byte: len(mask_var_str_code)]

                code_variables.append(source[var_start_byte: var_end_byte])
            else:
                mask_var_str_code = mask_var_str_code[0: var_start_byte] + \
                                    args.string_replaced_token + \
                                    mask_var_str_code[var_end_byte: len(mask_var_str_code)]

                mask_str_code = mask_str_code[0: var_start_byte] + \
                                args.string_replaced_token + \
                                mask_str_code[var_end_byte: len(mask_str_code)]

                code_strings.append(source[var_start_byte: var_end_byte])

    code_variables.reverse()
    code_strings.reverse()

    return mask_var_str_code, mask_str_code, code_variables, code_strings


def extract_function_call_from_code(source, lang):
    """
    Args:
        source: :(str) Source code string
        lang: :(str) Source code language
    Returns:
    """
    mask_function_call_str_code = copy.deepcopy(source)
    mask_str_code = copy.deepcopy(source)
    root = parse_ast(source=source, lang=lang)

    assert root is not None and "ERROR" not in root.sexp(), f"code pre-processed error"

    # extract variables
    identifier_nodes = []
    if lang in PATTERNS_IDENTIFIER:
        query = LANGUAGE[lang].query(PATTERNS_IDENTIFIER[lang])
        captures = query.captures(root)
        identifier_nodes = [get_node_position(node=capture[0], lang=lang) for capture in captures]

    # deduplication
    identifier_nodes = list(set(identifier_nodes))

    # extract attributes
    attr_nodes = []
    if lang in PATTERNS_ATTRIBUTE:
        query = LANGUAGE[lang].query(PATTERNS_ATTRIBUTE[lang])
        captures = query.captures(root)
        attr_nodes = [get_node_position(node=capture[0], lang=lang) for capture in captures]

    # deduplication
    attr_nodes = list(set(attr_nodes))

    # extract strings
    string_nodes = []
    if lang in PATTERNS_STRING:
        query = LANGUAGE[lang].query(PATTERNS_STRING[lang])
        captures = query.captures(root)
        string_nodes = [get_node_position(node=capture[0], lang=lang) for capture in captures]

    # deduplication
    string_nodes = list(set(string_nodes))

    _, head_node_end_byte = get_head_position(root, lang)
    assert not head_node_end_byte == 0, f"parse code head fail"

    # remove error tokens
    identifier_nodes_copy = copy.deepcopy(identifier_nodes)

    for str_start_byte, str_end_byte in identifier_nodes_copy:
        for sub_str_start_byte, sub_str_end_byte in identifier_nodes_copy:
            if (sub_str_start_byte >= str_start_byte and sub_str_end_byte < str_end_byte) or (
                    sub_str_start_byte > str_start_byte and sub_str_end_byte <= str_end_byte):
                if (sub_str_start_byte, sub_str_end_byte) in identifier_nodes:
                    identifier_nodes.remove((sub_str_start_byte, sub_str_end_byte))

    attr_nodes_copy = copy.deepcopy(attr_nodes)

    for str_start_byte, str_end_byte in attr_nodes_copy:
        for sub_str_start_byte, sub_str_end_byte in attr_nodes_copy:
            if (sub_str_start_byte >= str_start_byte and sub_str_end_byte < str_end_byte) or (
                    sub_str_start_byte > str_start_byte and sub_str_end_byte <= str_end_byte):
                if (sub_str_start_byte, sub_str_end_byte) in attr_nodes:
                    attr_nodes.remove((sub_str_start_byte, sub_str_end_byte))

    useful_string_node = copy.deepcopy(string_nodes)

    for str_start_byte, str_end_byte in string_nodes:
        for sub_str_start_byte, sub_str_end_byte in string_nodes:
            if (sub_str_start_byte >= str_start_byte and sub_str_end_byte < str_end_byte) or (
                    sub_str_start_byte > str_start_byte and sub_str_end_byte <= str_end_byte):
                if (sub_str_start_byte, sub_str_end_byte) in useful_string_node:
                    useful_string_node.remove((sub_str_start_byte, sub_str_end_byte))

    # remove identifiers in strings
    # useful_string_node = copy.deepcopy(string_nodes_wo_nest)

    identifier_nodes_copy = copy.deepcopy(identifier_nodes + attr_nodes)
    useful_string_node_copy = copy.deepcopy(useful_string_node)
    for str_start_byte, str_end_byte in useful_string_node_copy:
        for id_start_byte, id_end_byte in identifier_nodes_copy:
            if id_start_byte >= str_start_byte and id_end_byte <= str_end_byte:
                if (id_start_byte, id_end_byte) in attr_nodes:
                    attr_nodes.remove((id_start_byte, id_end_byte))

            if str_start_byte >= id_start_byte and str_end_byte <= id_end_byte:
                if (str_start_byte, str_end_byte) in useful_string_node:
                    useful_string_node.remove((str_start_byte, str_end_byte))

    removed_code_segments = attr_nodes + useful_string_node
    removed_code_segments.sort(key=operator.itemgetter(0), reverse=True)

    code_function_calls = []
    code_strings = []

    for (var_start_byte, var_end_byte) in removed_code_segments:
        # keep the head of a function
        if var_start_byte >= head_node_end_byte:

            if (var_start_byte, var_end_byte) in attr_nodes:

                mask_function_call_str_code = mask_function_call_str_code[0: var_start_byte] + \
                                              args.var_replaced_token + \
                                              mask_function_call_str_code[
                                              var_end_byte: len(mask_function_call_str_code)]

                code_function_calls.append(source[var_start_byte: var_end_byte])
            else:
                mask_function_call_str_code = mask_function_call_str_code[0: var_start_byte] + \
                                              args.string_replaced_token + \
                                              mask_function_call_str_code[
                                              var_end_byte: len(mask_function_call_str_code)]

                mask_str_code = mask_str_code[0: var_start_byte] + \
                                args.string_replaced_token + \
                                mask_str_code[var_end_byte: len(mask_str_code)]

                code_strings.append(source[var_start_byte: var_end_byte])

    code_function_calls.reverse()
    code_strings.reverse()

    return mask_function_call_str_code, mask_str_code, code_function_calls, code_strings


def extract_identifier_from_code(source, lang):
    """
    Args:
        source: :(str) Source code string
        lang: :(str) Source code language
    Returns:

    """
    mask_identifier_str_code = copy.deepcopy(source)
    mask_str_code = copy.deepcopy(source)
    root = parse_ast(source=source, lang=lang)

    assert root is not None and "ERROR" not in root.sexp(), f"code pre-processed error"

    # extract variables
    identifier_nodes: list[tuple] = []
    if lang in PATTERNS_IDENTIFIER:
        query = LANGUAGE[lang].query(PATTERNS_IDENTIFIER[lang])
        captures = query.captures(root)
        identifier_nodes = [get_node_position(node=capture[0], lang=lang) for capture in captures]

    # extract attributes
    attr_nodes = []
    if lang in PATTERNS_ATTRIBUTE:
        query = LANGUAGE[lang].query(PATTERNS_ATTRIBUTE[lang])
        captures = query.captures(root)
        attr_nodes = [get_node_position(node=capture[0], lang=lang) for capture in captures]

    # combine and remove repeat items
    identifier_nodes = identifier_nodes + attr_nodes
    identifier_nodes = list(set(identifier_nodes))

    # extract strings
    string_nodes = []
    if lang in PATTERNS_STRING:
        query = LANGUAGE[lang].query(PATTERNS_STRING[lang])
        captures = query.captures(root)
        string_nodes = [get_node_position(node=capture[0], lang=lang) for capture in captures]

    # deduplication
    string_nodes = list(set(string_nodes))

    _, head_node_end_byte = get_head_position(root, lang)
    assert not head_node_end_byte == 0, f"parse code head fail"

    # remove error tokens
    identifier_nodes_copy = copy.deepcopy(identifier_nodes)

    for str_start_byte, str_end_byte in identifier_nodes_copy:
        for sub_str_start_byte, sub_str_end_byte in identifier_nodes_copy:
            if (sub_str_start_byte >= str_start_byte and sub_str_end_byte < str_end_byte) or (
                    sub_str_start_byte > str_start_byte and sub_str_end_byte <= str_end_byte):
                if (sub_str_start_byte, sub_str_end_byte) in identifier_nodes:
                    identifier_nodes.remove((sub_str_start_byte, sub_str_end_byte))

    attr_nodes_copy = copy.deepcopy(attr_nodes)

    for str_start_byte, str_end_byte in attr_nodes_copy:
        for sub_str_start_byte, sub_str_end_byte in attr_nodes_copy:
            if (sub_str_start_byte >= str_start_byte and sub_str_end_byte < str_end_byte) or (
                    sub_str_start_byte > str_start_byte and sub_str_end_byte <= str_end_byte):
                if (sub_str_start_byte, sub_str_end_byte) in attr_nodes:
                    attr_nodes.remove((sub_str_start_byte, sub_str_end_byte))

    useful_string_node = copy.deepcopy(string_nodes)

    for str_start_byte, str_end_byte in string_nodes:
        for sub_str_start_byte, sub_str_end_byte in string_nodes:
            if (sub_str_start_byte >= str_start_byte and sub_str_end_byte < str_end_byte) or (
                    sub_str_start_byte > str_start_byte and sub_str_end_byte <= str_end_byte):
                if (sub_str_start_byte, sub_str_end_byte) in useful_string_node:
                    useful_string_node.remove((sub_str_start_byte, sub_str_end_byte))

    # useful_string_node = copy.deepcopy(string_nodes_wo_nest)

    # remove identifiers in strings
    identifier_nodes_copy = copy.deepcopy(identifier_nodes)
    useful_string_node_copy = copy.deepcopy(useful_string_node)
    for str_start_byte, str_end_byte in useful_string_node_copy:
        for id_start_byte, id_end_byte in identifier_nodes_copy:
            if id_start_byte >= str_start_byte and id_end_byte <= str_end_byte:
                if (id_start_byte, id_end_byte) in identifier_nodes:
                    identifier_nodes.remove((id_start_byte, id_end_byte))

            if str_start_byte >= id_start_byte and str_end_byte <= id_end_byte:
                if (str_start_byte, str_end_byte) in useful_string_node:
                    useful_string_node.remove((str_start_byte, str_end_byte))

    removed_code_segments = identifier_nodes + useful_string_node
    removed_code_segments.sort(key=operator.itemgetter(0), reverse=True)

    code_identifiers = []
    code_strings = []

    for (var_start_byte, var_end_byte) in removed_code_segments:
        # keep the head of a function
        if var_start_byte >= head_node_end_byte:

            if (var_start_byte, var_end_byte) in identifier_nodes:
                # deal with the special string replaced token we defined in the project
                # pay respect to the idea from tz
                mask_identifier_str_code = mask_identifier_str_code[0: var_start_byte] + \
                                           args.var_replaced_token + \
                                           mask_identifier_str_code[var_end_byte: len(mask_identifier_str_code)]

                code_identifiers.append(source[var_start_byte: var_end_byte])
            else:
                mask_identifier_str_code = mask_identifier_str_code[0: var_start_byte] + \
                                           args.string_replaced_token + \
                                           mask_identifier_str_code[var_end_byte: len(mask_identifier_str_code)]

                mask_str_code = mask_str_code[0: var_start_byte] + \
                                args.string_replaced_token + \
                                mask_str_code[var_end_byte: len(mask_str_code)]

                code_strings.append(source[var_start_byte: var_end_byte])

    code_identifiers.reverse()
    code_strings.reverse()

    return mask_identifier_str_code, mask_str_code, code_identifiers, code_strings


def extract_string_from_code(source, lang):
    """
    Args:
        source: :(str) Source code string
        lang: :(str) Source code language
    Returns:
        head (str): Definition of a function
        variables (str): All the variables in a function
        code_with_var_replaced (str): Source code with variables replaced by a special token
    """

    mask_str_code = copy.deepcopy(source)
    root = parse_ast(source=source, lang=lang)

    assert root is not None and "ERROR" not in root.sexp(), f"code pre-processed error"

    # extract variables
    identifier_nodes = []
    if lang in PATTERNS_IDENTIFIER:
        query = LANGUAGE[lang].query(PATTERNS_IDENTIFIER[lang])
        captures = query.captures(root)
        identifier_nodes = [get_node_position(node=capture[0], lang=lang) for capture in captures]

    # deduplication
    identifier_nodes = list(set(identifier_nodes))

    # extract attributes
    attr_nodes = []
    if lang in PATTERNS_ATTRIBUTE:
        query = LANGUAGE[lang].query(PATTERNS_ATTRIBUTE[lang])
        captures = query.captures(root)
        attr_nodes = [get_node_position(node=capture[0], lang=lang) for capture in captures]

    # deduplication
    attr_nodes = list(set(attr_nodes))

    # extract strings
    string_nodes = []
    if lang in PATTERNS_STRING:
        query = LANGUAGE[lang].query(PATTERNS_STRING[lang])
        captures = query.captures(root)
        string_nodes = [get_node_position(node=capture[0], lang=lang) for capture in captures]

    # deduplication
    string_nodes = list(set(string_nodes))

    _, head_node_end_byte = get_head_position(root, lang)
    assert not head_node_end_byte == 0, f"parse code head fail"

    # remove error tokens
    identifier_nodes_copy = copy.deepcopy(identifier_nodes)

    for str_start_byte, str_end_byte in identifier_nodes_copy:
        for sub_str_start_byte, sub_str_end_byte in identifier_nodes_copy:
            if (sub_str_start_byte >= str_start_byte and sub_str_end_byte < str_end_byte) or (
                    sub_str_start_byte > str_start_byte and sub_str_end_byte <= str_end_byte):
                if (sub_str_start_byte, sub_str_end_byte) in identifier_nodes:
                    identifier_nodes.remove((sub_str_start_byte, sub_str_end_byte))

    attr_nodes_copy = copy.deepcopy(attr_nodes)

    for str_start_byte, str_end_byte in attr_nodes_copy:
        for sub_str_start_byte, sub_str_end_byte in attr_nodes_copy:
            if (sub_str_start_byte >= str_start_byte and sub_str_end_byte < str_end_byte) or (
                    sub_str_start_byte > str_start_byte and sub_str_end_byte <= str_end_byte):
                if (sub_str_start_byte, sub_str_end_byte) in attr_nodes:
                    attr_nodes.remove((sub_str_start_byte, sub_str_end_byte))
    useful_string_node = copy.deepcopy(string_nodes)

    for str_start_byte, str_end_byte in string_nodes:
        for sub_str_start_byte, sub_str_end_byte in string_nodes:
            if (sub_str_start_byte >= str_start_byte and sub_str_end_byte < str_end_byte) or (
                    sub_str_start_byte > str_start_byte and sub_str_end_byte <= str_end_byte):
                if (sub_str_start_byte, sub_str_end_byte) in useful_string_node:
                    useful_string_node.remove((sub_str_start_byte, sub_str_end_byte))

    # remove identifiers in strings
    # remove strings in variables
    identifier_nodes_copy = copy.deepcopy(identifier_nodes + attr_nodes)
    useful_string_node_copy = copy.deepcopy(useful_string_node)
    for str_start_byte, str_end_byte in useful_string_node_copy:
        for id_start_byte, id_end_byte in identifier_nodes_copy:
            if id_start_byte >= str_start_byte and id_end_byte <= str_end_byte:
                if (id_start_byte, id_end_byte) in identifier_nodes:
                    identifier_nodes.remove((id_start_byte, id_end_byte))

            # e.g. foreach (static::${'_'.$rel_name} as $key => $settings)   in php language
            if str_start_byte >= id_start_byte and str_end_byte <= id_end_byte:
                if (str_start_byte, str_end_byte) in useful_string_node:
                    useful_string_node.remove((str_start_byte, str_end_byte))

    removed_code_segments = useful_string_node
    removed_code_segments.sort(key=operator.itemgetter(0), reverse=True)

    code_strings = []

    for (var_start_byte, var_end_byte) in removed_code_segments:
        # keep the head of a function
        if var_start_byte >= head_node_end_byte:
            mask_str_code = mask_str_code[0: var_start_byte] + \
                            args.string_replaced_token + \
                            mask_str_code[var_end_byte: len(mask_str_code)]

            code_strings.append(source[var_start_byte: var_end_byte])

    code_strings.reverse()

    return mask_str_code, code_strings
