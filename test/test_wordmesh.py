#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 19:22:42 2018

@author: mukund
"""
import sys
import os

#print(os.path.join(os.path.dirname(os.getcwd()), 'wordmesh'))
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'wordmesh'))

from wordmesh import StaticWordmesh, equilibrium_positions
import unittest

test_text = """Ground Control to Major Tom
Ground Control to Major Tom
Take your protein pills and put your helmet on
Ground Control to Major Tom (ten, nine, eight, seven, six)
Commencing countdown, engines on (five, four, three)
Check ignition and may God's love be with you (two, one, liftoff)
This is Ground Control to Major Tom
You've really made the grade
And the papers want to know whose shirts you wear
Now it's time to leave the capsule if you dare
"This is Major Tom to Ground Control
I'm stepping through the door
And I'm floating in a most peculiar way
And the stars look very different today
For here
Am I sitting in a tin can
Far above the world
Planet Earth is blue
And there's nothing I can do
Though I'm past one hundred thousand miles
I'm feeling very still
And I think my spaceship knows which way to go
Tell my wife I love her very much she knows
Ground Control to Major Tom
Your circuit's dead, there's something wrong
Can you hear me, Major Tom?
Can you hear me, Major Tom?
Can you hear me, Major Tom?
Can you "Here am I floating 'round my tin can
Far above the moon
Planet Earth is blue
And there's nothing I can do"
"""

class TestWordmesh(unittest.TestCase):
    
    def test_static_default_constructor(self):
        wm = StaticWordmesh(test_text)
        self.assertEqual(['planet', 'ground', 'countdown', 'knows', 'tom'], 
                         wm.keywords)
        
if __name__ == '__main__':
    #unittest.main()
    wm = StaticWordmesh(test_text, pos_filter=('JJ'))
    print(wm.keywords)
        
        
        
        