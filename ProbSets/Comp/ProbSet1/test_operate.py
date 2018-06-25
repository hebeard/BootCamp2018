# test_operate.py

import operate as op
import pytest

def test_typeError():
    with pytest.raises(TypeError) as excinfo:
        op.operate(1,2,3)
    assert excinfo.value.args[0] == "oper must be a string"
    
def test_add():
    assert op.operate(8,4,"+") == 12, "failed on '+' operation"
    
def test_sub():
    assert op.operate(8,4,"-") == 4, "failed on '-' operation"
    
def test_mul():
    assert op.operate(8,4,"*") == 32, "failed on '*' operation"
    
def test_truediv():
    assert op.operate(8,4,"/") == 2, "failed on '/' operation"
    
def test_zeroDivisionError():
    with pytest.raises(ZeroDivisionError) as excinfo:
        op.operate(1,0,"/")
    assert excinfo.value.args[0] == "division by zero is undefined"
    
def test_valueError():
    with pytest.raises(ValueError) as excinfo:
        op.operate(1,2,"hello")
    assert excinfo.value.args[0] == "oper must be one of '+', '/', '-', or '*'"