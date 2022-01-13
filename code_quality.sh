#!/bin/bash
cq_result=$(radon cc --min E -e *pyx dessia_common)
echo $cq_result
if [[ "$cq_result" ]];
  then 
	  echo "Error in code quality check, run radon to simplify functions">&2;
	  exit 64;
	
fi;
nb_pydoc_errors=$(pydocstyle --count --ignore D400,D415,D404 dessia_common *.py | tail -1)
echo $nb_pydoc_errors
if [[ "$nb_pydoc_errors" -gt 680 ]];
  then 
	  echo "Error in doc quality check, run pydocstyle to correct docstrings">&2;
	  exit 64;
fi;


