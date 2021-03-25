# PB (12-01-2021): credit for this code goes to Olha Zolotarenko, who wrote her BA thesis on the subject of coherence visualisation. This is just the conversion script to accomodate the format brat (which the viz component was based on) expects. (private repo to be found here: https://github.com/zolotarenko/VisualizationOfCoherenceRelations)

# Algorithm
# Idea: internal.json ---> parse ---> exetrnal_brat.json ---> load in index.xhtml as variable
# 1. DocData
# - argument&connective spans
# - minimal.txt --> text
# ////////////////
# var docData = {
#     // Our text of choice
#     text     : "blablabla. The snow was heavy. But we kept going. Until the sun came back. more blabla.",
#     // The entities entry holds all entity annotations
#     entities : [
#         /* Format: [${ID}, ${TYPE}, [[${START}, ${END}]]]
#             note that range of the offsets are [${START},${END}) */
#         ['T1', 'Arg1', [[11, 30]]],
#         ['T2', 'Arg2', [[35, 49]]],
#         ['T3', 'Connective', [[31, 34]]],
#         ['T4', 'Arg1', [[31, 49]]],
#         ['T5', 'Arg2', [[56, 74]]],
#         ['T6', 'Connective', [[50, 55]]],
#     ],
#     relations : [
#     // Format: [${ID}, ${TYPE}, [[${ARGNAME}, ${TARGET}], [${ARGNAME}, ${TARGET}]]]
#     ['R1', 'Comparison.Contrast', [['1', 'T1'], ['2', 'T2']]],
#     ['R2', 'Temporal.Asynchronous', [['3', 'T4'], ['4', 'T5']]]
#     ],
# };
# ////////////////


import json
from collections import OrderedDict
import re

def convert(textinput, jsoninput):
    
    docData = OrderedDict()

    #with open("/Users/apple/ba_visualization/data/olha_example_input.txt") as f:
    #with open(textinput) as f:
        #text = f.readlines()
    #text = textinput.split('\n')
    #text_line = ' '.join([line.strip() for line in text])
    #docData["text"] = text_line
    text = re.sub('\n', ' ', textinput)
    docData["text"] = text

    docData["entities"] = []
    docData["relations"] = []
    relations_counter_id = 1
    entities_counter_id = 1
    argname_counter = 1

    # Add our sentence/text to visualize


    def relation_spans():

        relation_pairs = []
        span_outgoing = []
        span_target = []
        for i in docData["entities"]:
            if i[1] == "Arg1":
                span_outgoing.append(i[0])
            elif i[1] == "Arg2":
                span_target.append(i[0])

        for x, y in zip(span_outgoing, span_target):
            pair = (x, y)
            relation_pairs.append(pair)

        return relation_pairs
    
    input_data = json.loads(jsoninput)
    entities = []


    # SECTION Build docData["entities"]
    # First level of iteration - every relation in the CoNLL2016 JSON format
    # E.g. ID : 1,2, ... , n
    for rel in input_data:
        # Iterating properties of very relation : key-value pairs in the relation
        # E.g. "ID", "DocID", "Sense" , ...
        for rel_property_key, rel_property_value in rel.items():
            # Arguments and connectives have dict values with needed "CharacterSpanList" in it
            # That's why here is the second iteration:
            # over the key-value pairs of the "Arg1", "Arg2" and "Connective"
            if type(rel_property_value) == dict:
                for entity_key, entity_value in rel_property_value.items():
                    if entity_key == 'CharacterSpanList':
                        # Possible values for "CharacterSpanList":
                        # - [[300, 372]]] - normal case, simple continuous argument/connective
                        # - [[405, 457], [543, 590]] - discontinuous argument/connective
                        # - [] - empty argument/connective for implicit relations or

                        # WARNING Fix of the bug (22.3.21):
                        # the tested document contains empty arguments, which was filtered earlier
                        # and was causing the wrong alignment of the relations
                        # if entity_value:

                        # Create entity and add to the docData["entities"]
                        entity = ["E" + str(entities_counter_id), rel_property_key, entity_value]
                        docData["entities"] += [entity]
                        # Increase the entity counter id for entity enumeration
                        entities_counter_id += 1

    # SECTION GET TYPES
    relation_type = []
    for rel in input_data:
        for rel_property_key, rel_property_value in rel.items():
            if rel_property_key == "Type":
                relation_type.append(rel_property_value)

    # SECTION GET SENSES
    for rel in input_data:
        for rel_property_key, rel_property_value in rel.items():
            #  Get the senses for relations
            if rel_property_key == "Sense":
                # Could be a random name but lets keep it as a number
                name = str(relation_type[relations_counter_id - 1]) + '.' + rel_property_value

                # relation_spans = relation_spans()
                relation = [
                    "R" + str(relations_counter_id),
                    name,
                    [
                        [str(argname_counter),
                         relation_spans()[argname_counter - 1][0]
                         ],
                        [str(argname_counter + 1),
                         relation_spans()[argname_counter - 1][1]
                         ]
                    ]
                ]

                # Format: ['R1', 'REL1', [['1', 'T1'], ['2', 'T2']]],
                docData["relations"] += [relation]

                relations_counter_id += 1
                argname_counter += 1
                
    return json.dumps(docData, ensure_ascii=False)
