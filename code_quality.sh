#!/bin/bash
cq_result=$(radon cc --min E -e *pyx dessia_common)
echo $cq_result
if [[ "$cq_result" ]];
  then 
	  echo "Error in code quality check, run radon to simplify functions">&2;
	  exit 64;
	
fi;

pydoc_result=$(pydocstyle --count --ignore D400,D415,D404 dessia_common *.py)
echo $pydoc_result
if [[ "$pydoc_result" ]];
  then 
	  echo "Error in doc quality check, run pydocstyle to correct docstrings">&2;
	  exit 64;
	
fi;


