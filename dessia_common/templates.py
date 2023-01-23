""" Templates for dessia_common. """
from string import Template

dessia_object_markdown_template = Template('''
# Object $name of class $class_

This is a markdown file https://www.markdownguide.org/cheat-sheet/

The good practice is to create a string python template and move the template to another python module
(like templates.py) to avoid mix python code and markdown, as python syntax conflicts with markdown

You can substitute values with object attributes like the name of the object: $name

# Attributes

Object $name of class $class_ has the following attributes:

$table

''')

workflow_template = Template('''
<!DOCTYPE html>
<html>
<head>
	<title>Dessia Workflow</title>
    <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/jointjs/2.1.0/joint.css" />
</head>
<body>
    <!-- content -->
    <div id="myholder" ></div>

    <!-- dependencies -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.4.1/jquery.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/lodash.js/3.10.1/lodash.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/backbone.js/1.4.0/backbone.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jointjs/3.1.1/joint.js"></script>

    <!-- code -->
    <script type="text/javascript">
            var workflow_data = $workflow_data;
			var blocks_data = workflow_data['blocks'];
			var edges_data = workflow_data['edges'];
			var nonblock_variables_data = workflow_data['nonblock_variables'];


			var graph = new joint.dia.Graph;
			// new joint.dia.Paper({ el: $$('#paper-create'), width: 650, height: 200, gridSize: 1, model: graph });
			var paper = new joint.dia.Paper({
          el: document.getElementById('myholder'),
          model: graph,
          width: '100%',
          height: 900,
          gridSize: 25,
					drawGrid: true,
      });


	    paper.options.defaultConnector = {
	      name: 'jumpover',
	      args: {
	        radius: 5
	      }
	    };

	    paper.options.defaultRouter = {
	      name: 'manhattan',
	      args: {
	        padding: 25
	      }
	    };

	    joint.dia.Link.define('standard.Link', {
	      attrs: {
	        line: {
	          connection: true,
	          stroke: '#333',
	          strokeWidth: 2,
	          strokeLinejoin: 'round',
	          targetMarker: {
	            'type': 'path',
	            'd': 'M 10 -5 0 0 10 5 z'
	            }
	          },
	          wrapper: {
	            connection: true,
	            strokeWidth: 10,
	            strokeLinejoin: 'round'
	            }
	          }
	      }, {
	      markup: [{
	        tagName: 'path',
	        selector: 'wrapper',
	        attributes: {
	          'fill': 'none',
	          'cursor': 'pointer',
	          'stroke': 'transparent'
	          }
	        }, {
	        tagName: 'path',
	        selector: 'line',
	        attributes: {
	          'fill': 'none',
	          'pointer-events': 'none'
	          }
	        }]
	      });


			var blocks = [];
			for (const [i, block_data] of blocks_data.entries()){

				var block = new joint.shapes.devs.Model({
						position: { x: block_data['position'][0],
						 						y: block_data['position'][1]},
						size: { width: 220, height: 120 },
						ports: {
								groups: {
										'in': {
											label: {
					                    position: {
					                        name: 'right', args: {
					                            x: 12,
																			y: 5,
					                        }}},
												attrs: {
														'.port-label': {fontSize: 8},
														'.port-body': {
															fill: '#DDD',
															stroke: 'black',
															strokeWidth: 2,
															height: 10,
															width: 10,
															magnet: 'passive'
														}
												},
												markup: '<rect class="port-body"/>'
										},
										'out': {
											label: {
					                    // label layout definition:
					                    position: {
					                        name: 'left', args: {
					                            x: -12,
																			y: 5,
					                        }}},
												attrs: {
														'.port-label': {fontSize: 8},
														'.port-body': {
																fill: '#CCC',
																stroke: 'black',
																strokeWidth: 2,
																height: 10,
																width: 10,
																magnet: true,
																'ref-x': -10
														}
												},
												markup: '<rect class="port-body"/>'
										}
								}
						},
						attrs: {
								'.label': { text: block_data["name"],
								 					'ref-x': .5, 'ref-y': .05,
													'font-size': 12 },
								rect: { fill: '#EEE' }
						}
				});

				// var input_ports = [];
				let j = 0;
				for (port_input_data of block_data['inputs']){
					var port = {
						id: String(i)+String(j),
		        group: 'in',
		        args: {},
		        attrs: { text: { text: port_input_data['name'] } },
		    	};
					block.addPort(port);
					if (port_input_data['is_workflow_input']){
						if (port_input_data['has_default_value']){
							block.portProp(port, 'attrs/.port-body/fill', '#FFA500');
						}
						else {
							block.portProp(port, 'attrs/.port-body/fill', '#00FF00');
						}
					}

					j++;
				}

				for (port_output_data of block_data['outputs']){
					var port = {
						id: String(i)+String(j),
		        group: 'out',
		        args: {},
		        attrs: { text: { text: port_output_data['name'] } },
		    	};
					block.addPort(port);
					if (port_output_data['is_workflow_output']){
						block.portProp(port, 'attrs/.port-body/fill', '#ff1919');
					}
					j++;
				}


			graph.addCell(block);
			blocks.push(block);
		}

		var nonblock_variables = []
		for (nonblock_variable_data of nonblock_variables_data){
			var variable = new joint.shapes.devs.Model({
					position: {x: nonblock_variable_data['position'][0],
						 				 y: nonblock_variable_data['position'][1]},
					size: { width: 100, height: 20 },
					attrs: {
							'.label': { text: nonblock_variable_data["name"],
												'ref-x': .5, 'ref-y': .05,
												'font-size': 12 },
							rect: { fill: '#EEE' }
					}
			});


		graph.addCell(variable);
		nonblock_variables.push(variable);
		}


		for (edge of edges_data){
			var node1 = edge[0];
			if (typeof(node1) == 'number'){
				var source = {'id': nonblock_variables[node1]};
			}
			else{
				var block1 = blocks[node1[0]];
				var port1 = block1.getPorts()[node1[1]];
				var source = {
					id: block1.id,
					port: port1.id
				}
			}

			var node2 = edge[1];
			if (typeof(node2) == 'number'){
				var target = {'id': nonblock_variables[node2]};
			}
			else{
				var block2 = blocks[node2[0]];
				var port2 = block2.getPorts()[edge[1][1]]
				var target = {
					id: block2.id,
					port: port2.id
				}
			}

			var link = new joint.shapes.standard.Link({
			      source: source,
			      target: target
			    });
			// Assume graph has the srcModel and dstModel with in and out ports.
			graph.addCell(link)
		}



    </script>
</body>
</html>
''')


visjs_template = Template('''
<html>
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


dataset_markdown_template = Template('''
# Dataset $name of $element_details:

$table

## Information:

    This is a standard markdown: https://www.markdownguide.org/cheat-sheet/

    The good practice is to create a string python template and move the template to another python module
    (like templates.py) to avoid mix python code and markdown, as python syntax conflicts with markdown

    You can substitute values with object attributes like the name of the object: $name

''')
