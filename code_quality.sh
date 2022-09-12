#!/bin/bash

max_pydoc_errors=223

cq_result=$(radon cc --min F -e *pyx dessia_common)
echo $cq_result
if [[ "$cq_result" ]];
  then 
	  echo "Error in code quality check, run radon to simplify functions">&2;
	  exit 64;
	
fi;

nb_pydoc_errors=$(pydocstyle --count dessia_common/*.py | tail -1)
echo "$nb_pydoc_errors pydoc errors, limit is $max_pydoc_errors"
if [[ "$nb_pydoc_errors" -gt "$max_pydoc_errors" ]];
  then 
	  echo "Error in doc quality check, run pydocstyle to correct docstrings">&2;
	  exit 64;
  else
	  echo "You can lower number of pydoc errors to $nb_pydoc_errors (actual $max_pydoc_errors)"
fi;


