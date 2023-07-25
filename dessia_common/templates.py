""" Templates for dessia_common. """
from string import Template

dessia_object_markdown_template = Template('''# Object $name of class $class_

This is a markdown file https://www.markdownguide.org/cheat-sheet/

The good practice is to create a string python template and move the template to another python module
(like templates.py) to avoid mix python code and markdown, as python syntax conflicts with markdown

You can substitute values with object attributes like the name of the object: $name

# Attributes

Object $name of class $class_ has the following attributes:

$table

''')


visjs_template = Template('''<html>
        <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" />
        <meta charset="utf-8"/>
        <style type="text/css">
            #mynetwork {
                border: 1px solid lightgray;
            }
        </style>
        <title>$name</title>
    </head>
    <body>
    <div id="mynetwork"></div>

    <script type="text/javascript">
    var nodes = new vis.DataSet($nodes);

var edges = new vis.DataSet($edges);


// create a network
    var container = document.getElementById('mynetwork');

    // provide the data in the vis format
    var data = {
        nodes: nodes,
        edges: edges
    };

    scale=function (min,max,total,value) {
      if (max === min) {
        return 0.5;
      }
      else {
        var scale = 1 / (max);
        return Math.max(0,(value )*scale);
      }
    }

    var options = {edges: {scaling: {'customScalingFunction': scale}}};

    // initialize your network!
    var network = new vis.Network(container, data, options);
</script>
</body>
</html>
''')


dataset_markdown_template = Template('''# Dataset $name of $element_details:

$table

## Information:

    This is a standard markdown: https://www.markdownguide.org/cheat-sheet/

    The good practice is to create a string python template and move the template to another python module
    (like templates.py) to avoid mix python code and markdown, as python syntax conflicts with markdown

    You can substitute values with object attributes like the name of the object: $name

''')


workflow_state_markdown_template = Template('''# $class_ $name

progress $progress% on workflow $workflow_name

# Execution informations

$table

''')
