#!/bin/bash

max_pydoc_errors=413

cq_result=$(radon cc --min E -e *pyx dessia_common)
echo $cq_result
if [[ "$cq_result" ]];
  then 
	  echo "Error in code quality check, run radon to simplify functions">&2;
	  exit 64;
	
fi;

nb_pydoc_errors=$(pydocstyle --count --ignore D400,D415,D404,D212,D205,D200,D203,D401,D210 dessia_common/*.py | tail -1)
echo "$nb_pydoc_errors pydoc errors, limit is $max_pydoc_errors"
if [[ "$nb_pydoc_errors" -gt "$max_pydoc_errors" ]];
  then 
	  echo "Error in doc quality check, run pydocstyle to correct docstrings">&2;
	  exit 64;
  else
	  echo "You can lower number of pydoc errors to $nb_pydoc_errors (actual $max_pydoc_errors)"
fi;


