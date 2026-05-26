# Copyright contributors to the EVA project
# Licensed under the MIT License.
# Original source: https://github.com/microsoft/EVA
#
# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



from .. import *

_current_program = None
def _curr():
    """ Returns the CeriumProgram that is currently in context """
    global _current_program
    if _current_program == None:
        raise RuntimeError("No Program in context")
    return _current_program

_current_function = None
def _curr_function():
    """ Returns the CeriumFunction that is currently in context """
    global _current_function
    if _current_function == None:
        raise RuntimeError("No Function in context")
    return _current_function

def _get_term(x, program):
    """ Maps supported types into terms """
    if isinstance(x, Expression):
        return x.term
    elif isinstance(x, Term):
        return x
    else:
        raise TypeError("No conversion to Term available for " + str(x))

class Expression():
    """ Wrapper for Cerium's Term class. Provides operator overloads that
        create terms in the associated CeriumProgram.
        
        Attributes
        ----------
        term
            The native term
        program :
            The program the wrapped term is in
        """

    def __init__(self, term, program, func):
        self.term = term
        self.program = program
        self.function = func

    def __add__(self,other):
        """ Create a new addition term """
        return Expression(self.function._make_add(self.term, _get_term(other, self.program)), self.program, self.function)

    def __radd__(self,other):
        """ Create a new addition term """
        return Expression(self.function._make_add(_get_term(other, self.program), self.term), self.program, self.function)

    def __sub__(self,other):
        """ Create a new subtraction term """
        return Expression(self.function._make_subtract(self.term, _get_term(other, self.program)), self.program, self.function)

    def __rsub__(self,other):
        """ Create a new subtraction term """
        return Expression(self.function._make_subtract(_get_term(other, self.program), self.term), self.program, self.function)

    def __mul__(self,other):
        """ Create a new multiplication term """
        return Expression(self.function._make_multiply(self.term, _get_term(other, self.program)), self.program, self.function)

    def __rmul__(self,other):
        """ Create a new multiplication term """
        return Expression(self.function._make_multiply(_get_term(other, self.program), self.term), self.program, self.function)

    def __lshift__(self,rotation):
        """ Create a left rotation term """
        return Expression(self.function._make_left_rotation(self.term, rotation), self.program, self.function)

    def __rshift__(self,rotation):
        """ Create a right rotation term """
        return Expression(self.function._make_right_rotation(self.term, rotation), self.program, self.function)
    
    def __neg__(self):
        """ Create a negation term """
        return Expression(self.function._make_negate(self.term), self.program, self.function)

    def rescale(self):
        """ Create a rescale term """
        return Expression(self.function._make_rescale(self.term), self.program, self.function)

    def doubleRescale(self):
        """ Create a double rescale term """
        return Expression(self.function._make_double_rescale(self.term), self.program, self.function)

    def relinearize(self, rescaleLevels=0):
        """ Create a reliniearization term """
        return Expression(self.function._make_relinearize(self.term, rescaleLevels), self.program, self.function)

    def relinearize2(self):
        """ Create a reliniearization term """
        return Expression(self.function._make_relinearize2(self.term), self.program, self.function)

    def relinearize3(self):
        """ Create a reliniearization term """
        return Expression(self.function._make_relinearize3(self.term), self.program, self.function)

    def toEphemeral(self):
        """ Create a term encrypted with the ephemeral key """
        return Expression(self.function._make_ephemeral(self.term), self.program, self.function)

    def modswitch(self):
        """ Create a reliniearization term """
        return Expression(self.function._make_modswitch(self.term), self.program, self.function)

    def bootstrapModRaise(self, newLevel=0):
        """ Create a bootstrap modraise term """
        return Expression(self.function._make_bootstrap_modraise(self.term, newLevel), self.program, self.function)

    def conjugate(self):
        """ Create a conjugation term """
        return Expression(self.function._make_conjugate(self.term), self.program, self.function)

    def conjugate2(self):
        """ Create a conjugation term """
        return Expression(self.function._make_conjugate2(self.term), self.program, self.function)

    def rotate2(self,rotation):
        """ Create a rotation term """
        return Expression(self.function._make_rotate2(self.term, rotation), self.program, self.function)

    def rotate3(self,rotation):
        """ Create a rotation term """
        return Expression(self.function._make_rotate3(self.term, rotation), self.program, self.function)
    
    def level(self):
        """ Level of this term """
        return self.term.level

    def scale(self):
        """ Scale of this term """
        return self.term.scale

class CeriumProgram(Program):
    """ A wrapper for the native Program class. Acts as a context manager to
        set the program the Input and Output free functions operate on. """

    def __init__(self, name, rns_bit_size=28, num_gpus=1):
        """ Create a new CeriumProgram with a name and a vector size
            
            Parameters
            ----------
            name : str
                The name of the program
            rns_bit_size: int
                The size in bits of the RNS primes
            num_gpus: int
                The number of gpus to be compiled for
            """
        super().__init__(name, rns_bit_size, num_gpus)

    def __enter__(self):
        global _current_program
        if _current_program != None:
            raise RuntimeError("There is already a Program in context")
        _current_program = self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _current_program
        if _current_program != self:
            raise RuntimeError("This program is not currently in context")
        _current_program = None

class CeriumFunction(Function):
    """ A wrapper for the native Function class. Acts as a context manager to
        set the program the Input and Output free functions operate on. """

    def __init__(self, name, partitionSize, partitionId):
        """ Create a new CeriumFunction with a name
            
            Parameters
            ----------
            name : str
                The name of the program
            """
        program = _curr()
        super().__init__(name, program, partitionSize, partitionId)
        self._add_function_to_program()


    def __enter__(self):
        global _current_function
        if _current_function != None:
            raise RuntimeError("There is already a Function in context")
        _current_function = self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _current_function
        if _current_function != self:
            raise RuntimeError("This function is not currently in context")
        _current_function = None


def PlaintextInput(name, scale, level, scalar=False, remapable=False):
    """ Create a new named plaintext input term in the current CeriumProgram

        Parameters
        ----------
        name : str
            The name of the input
        scale: int
            The scale of the input
        level: int 
            The level of the input
        scalar: bool
            Whether the input is a scalar value
        remapable: bool 
            Whether the plaintext value can be remaped
        """
    program = _curr()
    function = _curr_function()
    return Expression(function._make_plaintext_input(name, scale, level, scalar, remapable), program, function)

def PeriodicPlaintextInput(name, scale, level, period, remapable=False):
    """ Create a new named periodic plaintext term in the current CeriumProgram that implements Cerium's Plaintext Compression Algorithm

        Parameters
        ----------
        name : str
            The name of the input
        scale: int
            The scale of the input
        level: int 
            The level of the input
        period: int 
            The power of 2 size of the repeated plaintext input
        remapable: bool 
            Whether the plaintext value can be remaped
        """
    program = _curr()
    function = _curr_function()
    return Expression(function._make_repeated_plaintext_input(name, scale, level, period, remapable), program, function)


def CiphertextInput(name, scale, level):
    """ Create a new named input term in the current CeriumProgram

        Parameters
        ----------
        name : str
            The name of the input
        scale: int 
            The scale of the input
        level: int
            The level of the input
        """
    program = _curr()
    function = _curr_function()
    return Expression(function._make_ciphertext_input(name, scale, level), program, function)

def CiphertextArgument(name, scale, level):
    """ Create a new named ciphertext argument term in the current CeriumFunction

        Parameters
        ----------
        name : str
            The name of the input
        scale: int 
            The scale of the input
        level: int
            The level of the input
        """
    program = _curr()
    func = _curr_function()
    return Expression(func._make_ciphertext_argument(name, scale, level), program, func)


def Receive(term):
    program = _curr()
    function = _curr_function()
    """ Create a new receive term """
    return Expression(function._make_receive(_get_term(term, program)), program, function)

def Partition(size, id):
    """ Create a new partition in the current CeriumProgram """
    program = _curr()
    function = _curr_function()
    return function._make_partition(size,id)

def CurrentPartitionSize():
    """ Get the Currently Set Partition Size """
    program = _curr()
    function = _curr_function()
    return function._currentPartitionSize()

def CurrentPartitionID():
    """ Get the Currently Set Partition ID """
    program = _curr()
    function = _curr_function()
    return function._currentPartitionID()

def CeriumStream(StreamSize,NumStreams,StreamFn,*argv,**kwargs):
    """ Create Parallel Execution Streams """
    currentPartitionSize = CurrentPartitionSize()
    currentPartitionID = CurrentPartitionID()
    if StreamSize < 1:
        raise Exception("Stream size must be greater than or equal to 1")
    if StreamSize > currentPartitionSize:
        raise Exception(f"Stream size must be less than currentPartitionSize={currentPartitionSize}")
    
    numParallelStreams = currentPartitionSize // StreamSize
    for sid in range(NumStreams):
        Partition(StreamSize,currentPartitionID*numParallelStreams + (sid % numParallelStreams))
        StreamFn(sid,*argv,**kwargs)
    Partition(currentPartitionSize,currentPartitionID)


def RotateMultiplyAccumulate(ciphertext, plaintexts, rotations):
    """ Create a Rotate Multiply Accumulate term """
    program = _curr()
    ciphertextTerm = _get_term(ciphertext, program)
    plaintextTerms = [_get_term(plaintext, program) for plaintext in plaintexts]
    function = _curr_function()
    return Expression(function._make_rotate_multiply_accumulate(ciphertextTerm, plaintextTerms, rotations), program, function)

def MultiplyRotateAccumulate(ciphertext, plaintexts, rotations):
    """ Create a Multiply Rotate Accumulate term """
    program = _curr()
    ciphertextTerm = _get_term(ciphertext, program)
    plaintextTerms = [_get_term(plaintext, program) for plaintext in plaintexts]
    function = _curr_function()
    return Expression(function._make_multiply_rotate_accumulate(ciphertextTerm, plaintextTerms, rotations), program, function)

def RotateAccumulate(ciphertext, rotations):
    """ Create a Rotate Accumulate term """
    program = _curr()
    ciphertextTerm = _get_term(ciphertext, program)
    function = _curr_function()
    return Expression(function._make_rotate_accumulate(ciphertextTerm, rotations), program, function)

def RotateAccumulateMany(ciphertexts, rotations):
    """ Create a Rotate Accumulate Many term """
    program = _curr()
    ciphertextTerms = [_get_term(ct, program) for ct in  ciphertexts]
    function = _curr_function()
    return Expression(function._make_rotate_accumulate_many(ciphertextTerms, rotations), program, function)

def HoistedRotate(ciphertext, rotations):
    """ Create a Hoisted Rotate term """
    program = _curr()
    ciphertextTerm = _get_term(ciphertext, program)
    function = _curr_function()
    terms = function._make_hoisted_rotate(ciphertextTerm, rotations)
    terms = [Expression(t, program, function) for t in terms]
    return terms

def BsgsMultiplyAccumulate(ciphertext, plaintexts, babySteprotations, giantStepRotations, rescaleLevels=0):
    """ Create a Baby Step Giant Step Multiply Accumulate term """
    program = _curr()
    function = _curr_function()
    ciphertextTerm = _get_term(ciphertext, program)
    plaintextTerms = [_get_term(plaintext, program) for plaintext in plaintexts]
    return Expression(function._make_bsgs_multiply_accumulate(ciphertextTerm, plaintextTerms, babySteprotations, giantStepRotations, rescaleLevels), program, function)

def MakeVector(values):
    """ Create a new ciphertext vector term """
    program = _curr()
    function = _curr_function()
    terms = [_get_term(value, program) for value in values]
    for t in terms:
        if t.scale != terms[0].scale:
            raise ValueError("All terms in a vector must have the same scale")
        if t.level != terms[0].level:
            raise ValueError("All terms in a vector must have the same level")
    return Expression(function._make_vector(terms), program, function)

def BreakVector(term):
    """ Break a ciphertext vector term """
    program = _curr()
    function = _curr_function()
    terms = function._make_extract_vector(_get_term(term, program))
    terms = [Expression(t, program, function) for t in terms]
    return terms

def Output(name, expr):
    """ Create a new named output term in the current CeriumProgram

        Parameters
        ----------
        name : str
            The name of the output
        """
    program = _curr()
    function = _curr_function()
    function._make_output(name, _get_term(expr, program))

def FunctionOutput(name, expr):
    """ Create a new named output term in the current CeriumProgram

        Parameters
        ----------
        name : str
            The name of the output
        """
    program = _curr()
    function = _curr_function()
    function._make_function_output(name, _get_term(expr, program))


def CeriumFunctionCall(function,arguments,remapableBase=""):
    """ Create a new named output term in the current CeriumProgram

        Parameters
        ----------
        name : str
            The name of the output
        """
    program = _curr()
    function_ = _curr_function()
    for k,v in arguments.items():
        arguments[k] =  _get_term(v, program)
    outputs = function_._make_function_call(function,arguments,remapableBase)
    for k,v in outputs.items():
        outputs[k] = Expression(v,program,function_)
    return outputs

