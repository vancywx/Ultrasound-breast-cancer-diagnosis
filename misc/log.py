'''
Coding:     UTF-8
Author:     Luo Luyang <luo960730@gmail.com, lyluo4@cse.cuhk.edu.hk>
Project:    CUHK Medical 2017 Summer
'''
from __future__ import print_function
import sys

class ProgressBar:
    '''
    Code Referred From : http://www.cnblogs.com/lustralisk/p/pythonProgressBar.html
    Make a moving bar for counting steps.
    '''
    def __init__(self, total = 0, width = 100):
        '''
        Args:
            total: integer, total num of iterations
            width: integer, length of the bar.
        '''
        self.count = 0
        self.total = total
        self.width = width
    def move(self):
        self.count += 1
    def log(self, job):
        '''Log out information.
        Args:
            job: String. Print the job you are doing. 
        '''
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        progress = self.width * self.count / self.total
        #sys.stdout.write(job + '{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('[' + '=' * int(progress) +'>'+ '.' * int(self.width - progress) + ']')
        sys.stdout.write(job + '  {0:3}/{1:3}: \r'.format(self.count, self.total))
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()

if __name__ == '__main__':
    import time
    for x in range(2):
        bar = ProgressBar(total = 5)
        for i in range(5):
            bar.move()
            bar.log( job = 'Gen')
            time.sleep(1)##
