Customizing Dessia Object default behaviors
===========================================


Serialization
-------------

DessiaObject implements out of the box:
 * serialization: transforming an object in a dictionary which can be dumped as json
 * deserialization: transforming back the dictionnary into a python object

Generic serialization parameters 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data equality can be controled by overloading the class attribute non_serializable_attributes
It creates a blacklist of attributes which should not be in the dictionary.


Overloading the to_dict method 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Be careful with this method! It enhance performance but needs to be updated when the code changes!

Overloading the to_dict method should:
 * Implement the parameters of the base class:

   * use_pointers: a boolean that indicates if pointers should be inserted in the dict
   * memo: a dict [pythonobject -> path] of already serialized values
   * path: the path in the global object: for most case append '/attribute_name' to given path for your attributes
 * Test if the object is in the memo if you want to use pointers
 * write the 'object_class' and 'name' in the dict for the dict to object: you can use DessiaObject.base_dict to do so.

If calling recursively to_dict of subattributes, these arguments must be passed, as is for use_pointers and memo. Path must be created as a subpath with '/' delimiter:
If you want the object to be reinserted as a pointer elsewhere, you should put it in the memo

.. code-block:: python
    
    def to_dict(self, use_pointers=True, memo=None, path='#')
        '''
        This is my specific to_dict!
        '''
        # Init memo is None
        if memo is None:
            memo = {}

        # If object is in the memo, inserting the pointer
        if use_pointers and self in memo:
            return {'ref__': memo[self]}

        # init dict
        dict_ = self.base_dict()

        # Putting in dict attribute1. Very bad name by the way.
        dict['attribute1'] = self.attribute1.to_dict(use_pointers=use_pointers,
                                                     memo=memo,
                                                     path=f'{path}/attribute1')
        
        # Putting in memo
        memo[self.attribute1] = f'{path}/attribute1'

        return dict_

Overloading the dict_to_object method 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Be also careful with this method!

Overloading the to_dict method should:
 * Implement the parameters of the base class:

   * global_dict: default to None.
   * pointers_memo: a dict [pythonobject -> path] of already serialized values
   * path: the path in the global object: for most case append '/attribute_name' to given path for your attributes

If calling recursively to_dict of subattributes, these arguments must be passed, as is for use_pointers and memo. Path must be created as a subpath with '/' delimiter:

.. code-block:: python
   
    @classmethod
    def dict_to_object(cls, dict_, global_dict=None, pointers_memo=None, path='#'):
        '''
        This is my specific dict to object
        '''
        # Init pointers memo
        if pointers_memo is None:
            pointers_memo = {}

        # If global dict is none, we should populate pointer memo with an analysis of dict_
        if global_dict is None:
            global_dict = dict_
            pointers_memo.update(dereference_jsonpointers(dict_))
                
        # getting attributes from dict. Very bad names by the way.
        attr1 = dict_['attr1']
        attr2 = dict_['attr2']
        # Instantiate
        return cls(attr1=attr1, attr2=attr2)




Object Equality
---------------


``__eq__`` and ``__hash__`` rule how objects are behaving whenever we test if one is equal to one another.


By default, Python use the method from type object (python base object) that only check for strict equalities.

It means that ``==`` and ``is`` method are equivalent, and checks for a strict equality, on object adress in computer memory.


Overwriting ``__eq__`` enables us to redefine ``==`` so that it is based on data.

It is important if we want to store involved object in our database, setting its class attribute ``_standalone_in_db`` to True.

As a matter of fact, MongoDB needs an equality on data to function properly.

It also has a conceptual meaning in our *Object Oriented Engineering* vision as, for instance, two bearings that have exactly same dimensions are, physically speaking, the same object.


A custom ``__eq__`` method needs a relevant ``__hash__`` are the two are working together.


A hash is an integer value that is equivalent to an identifier. Two objects that are equal, on a data level, *must* share the same hash. It is a necessary but not sufficient condition, as two objects with the same hash might not be equal.

The contraposition is that have different hashes are not equal.


DessiaObject defines generic ``__eq__`` and ``__hash__`` functions that are based on following class attributes : 

* ``_non_data_eq_attributes (['name'])`` 
* ``_non_data_hash_attributes (['name'])`` 


Any attribute listed in these sequence (by default, just DessiaObject's name) aren't taken into account for equalities.


