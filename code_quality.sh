#!/bin/bash

max_pydoc_errors=209

cq_result=$(radon cc --min F -e *pyx dessia_common)
echo $cq_result
if [[ "$cq_result" ]];
  then 
	  echo "Error in code quality check, run radon to simplify functions">&2;
	  exit 64;
	
fi;

