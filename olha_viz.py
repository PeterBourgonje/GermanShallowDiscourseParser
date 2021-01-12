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
    text = textinput.split('\n')
    text_line = '\t'.join([line.strip() for line in text]) # PB: not sure why text is joined on tabs here. Untested for viz component, may need fixing.
    docData["text"] = text_line

    docData["entities"] = []
    docData["relations"] = []
    relationsCounter_ID = 1
    entitiesCounter_ID = 1
    relationsARGNAME = 1
    argnameCounter = 1

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

        # print(span_outgoing, span_target)
        for x, y in zip(span_outgoing, span_target):
            pair = (x, y)
            relation_pairs.append(pair)

        return relation_pairs

    # Read internal json-file
    #with open("/Users/apple/ba_visualization/data/olha_example_output.json", "r") as f:
    #with open(jsoninput, "r") as f:
    #input_data = json.load(f)
    input_data = json.loads(jsoninput)
    entities = []



    for rel in input_data:
        for key1, value1 in rel.items():
            # Get the spans
            if type(value1) == dict:
                for key2, value2 in value1.items():
                    if key2 == 'CharacterSpanList':
                        #  Here we have to go -1 position as brat doesn't count as python from 0
                        # Could be: "CharacterSpanList": [[405, 457], [543, 590]] - discont
                        # Or: [] when implicit

                        if value2:
                            # entity = ["E" + str(entitiesCounter_ID), key1, [[value2[0][0]-1, value2[0][1]-1]]]
                            entity = ["E" + str(entitiesCounter_ID), key1, value2]
                            docData["entities"] += [entity]
                            entitiesCounter_ID += 1

    relation_type = []
    for rel in input_data:
        for key1, value1 in rel.items():
            if key1 == "Type":
                relation_type.append(value1)

    #print(relation_type)

    for rel in input_data:
        for key1, value1 in rel.items():
            #  Get the senses for relations
            # if key1 == "Type":
            #     relation_type = value1
            if key1 == "Sense":
                # Could be a random name but lets keep it a a number
                # target1 = [docData.get('entities')[i] for i in docData]
                name = str(relation_type[relationsCounter_ID-1]) + '.' + value1
                # target2 = [docData.get('entities')[0] for i in docData if docData.get('entities')[1] == 'Arg2'][0]
                relation = ["R" + str(relationsCounter_ID), name, [[str(argnameCounter), relation_spans()[argnameCounter-1][0]], [str(argnameCounter+1), relation_spans()[argnameCounter-1][1]]]]
                # Format: ['R1', 'REL1', [['1', 'T1'], ['2', 'T2']]],
                docData["relations"] += [relation]

                relationsCounter_ID += 1
                argnameCounter += 1

    #print(docData)


    # for rel in docData["relations"]:
    #     for

    # Write to data.json
    #with open("/Users/apple/ba_visualization/data/converted-types.json", "w") as f:
    #with open(re.sub('\..*', '', jsoninput) + '.visualisation.json', "w") as f:
        #json.dump(docData, f)
    return json.dumps(docData)

