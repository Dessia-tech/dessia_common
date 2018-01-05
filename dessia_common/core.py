#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 12:17:30 2018

@author: Steven Masfaraud masfaraud@dessia.tech
"""

import ZODB
import transaction
#import ZODB.config
from ZEO import client
import random


from BTrees.IOBTree import IOBTree

class ResultsDBClient:
    """
    Abstract class of Results client for ZODB Database
    """
    def __init__(self,address,model_name):
        self.address=address
        self.storage = client(address)
        self.db = ZODB.DB(self.storage)
        self.model_name
        #self.db = ZODB.config.databaseFromString(conf)
        
        connection=self.db.open()
        try:
            getattr(connection.root,self.name)
        except AttributeError:
            setattr(connection.root,self.name,IOBTree())
            transaction.commit()
        connection.close()
        
    def __del__(self):
        self.CloseDB()
        
    def CloseDB(self):
        self.db.close()
        
    def AddModel3DResult(self,result): 
        connection=self.db.open()
        try:
            mk=getattr(connection.root,self.name).maxKey()
            k=random.randint(0,mk+1)
            if k in getattr(connection.root,self.name):
                k=mk+1
        except ValueError:
            k=0

        getattr(connection.root,self.name)[k]=result
        transaction.commit()
        connection.close()
        return k
    
    def get_result(self,id_result):
        connection=self.db.open()
        result=getattr(connection.root,self.name)[id_result]
        connection.close()
        return result

    def _get_results(self):
        connection=self.db.open()
        results=dict(getattr(connection.root,self.name))
        connection.close()
        return results
    
