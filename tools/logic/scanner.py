import sys

"""
These classes have been taken from github.com/ashishgaurav13/wm2
"""

class Token(object):
    def __init__(self):
        self.kind = 0  # token kind
        self.pos = 0  # token position in the source text (starting at 0)
        self.col = 0  # token column (starting at 0)
        self.line = 0  # token line (starting at 1)
        self.val = u''  # token value
        self.next = None  # AW 2003-03-07 Tokens are kept in linked list
        self.until = None
        self.result = None
        self.index = 0

    def __str__(self):
        return "line {} col {} : {} {}".format(self.line, self.col, self.kind,
                                               self.val)


class Position(
        object
):  # position of source code stretch (e.g. semantic action, resolver expressions)
    def __init__(self, buf, beg, len, col):
        assert isinstance(buf, Buffer)
        assert isinstance(beg, int)
        assert isinstance(len, int)
        assert isinstance(col, int)

        self.buf = buf
        self.beg = beg  # start relative to the beginning of the file
        self.len = len  # length of stretch
        self.col = col  # column number of start position

    def getSubstring(self):
        return self.buf.readPosition(self)


class Buffer(object):
    EOF = u'\u0100'  # 256

    def __init__(self, s):
        self.buf = s
        self.bufLen = len(s)
        self.pos = 0
        self.lines = s.splitlines(True)

    def Read(self):
        if self.pos < self.bufLen:
            # result = unichr(ord(self.buf[self.pos]) & 0xff)   # mask out sign bits
            # Sean replaced unichr() with chr() for Python 3 compatibility
            result = chr(ord(self.buf[self.pos]) & 0xff)  # mask out sign bits
            self.pos += 1
            return result
        else:
            return Buffer.EOF

    def ReadChars(self, numBytes=1):
        result = self.buf[self.pos:self.pos + numBytes]
        self.pos += numBytes
        return result

    def Peek(self):
        if self.pos < self.bufLen:
            # return unichr(ord(self.buf[self.pos]) & 0xff)    # mask out sign bits
            # Sean replaced unichr() with chr() for Python 3 compatibility
            return chr(ord(self.buf[self.pos]) & 0xff)  # mask out sign bits
        else:
            return Scanner.buffer.EOF

    def getString(self, beg, end):
        s = ''
        oldPos = self.getPos()
        self.setPos(beg)
        while beg < end:
            s += self.Read()
            beg += 1
        self.setPos(oldPos)
        return s

    def getPos(self):
        return self.pos

    def setPos(self, value):
        if value < 0:
            self.pos = 0
        elif value >= self.bufLen:
            self.pos = self.bufLen
        else:
            self.pos = value

    def readPosition(self, pos):
        assert isinstance(pos, Position)
        self.setPos(pos.beg)
        return self.ReadChars(pos.len)

    def __iter__(self):
        return iter(self.lines)


class Scanner(object):
    EOL = u'\n'
    eofSym = 0

    charSetSize = 256
    maxT = 14
    noSym = 14
    start = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1
    ]

    def __str__(self):
        return self.buffer.buf

    def __init__(self, s):
        # self.buffer = Buffer( unicode(s) ) # the buffer instance
        # Sean removed unicode for Python 3.5 compatibility
        self.buffer = Buffer(s)  # the buffer instance

        self.ch = u'\0'  # current input character
        self.pos = -1  # column number of current character
        self.line = 1  # line number of current character
        self.lineStart = 0  # start position of current line
        self.oldEols = 0  # EOLs that appeared in a comment;
        self.NextCh()
        self.ignore = set()  # set of characters to be ignored by the scanner
        self.ignore.add(ord(' '))  # blanks are always white space
        self.ignore.add(9)
        self.ignore.add(10)
        self.ignore.add(13)

        # fill token list
        self.tokens = Token()  # the complete input token stream
        node = self.tokens

        node.next = self.NextToken()
        node = node.next
        while node.kind != Scanner.eofSym:
            if node.kind == self.noSym:
                print("{} : ({}, {}) unrecognized symbol".format(
                    self.buffer.buf, node.line, node.col))
                sys.exit(1)
            node.next = self.NextToken()
            node = node.next

        node.next = node
        node.val = u'EOF'
        self.t = self.tokens  # current token
        self.pt = self.tokens  # current peek token

    def NextCh(self):
        if self.oldEols > 0:
            self.ch = Scanner.EOL
            self.oldEols -= 1
        else:
            self.ch = self.buffer.Read()
            self.pos += 1
            # replace isolated '\r' by '\n' in order to make
            # eol handling uniform across Windows, Unix and Mac
            if (self.ch == u'\r') and (self.buffer.Peek() != u'\n'):
                self.ch = Scanner.EOL
            if self.ch == Scanner.EOL:
                self.line += 1
                self.lineStart = self.pos + 1

    def Comment0(self):
        level = 1
        line0 = self.line
        lineStart0 = self.lineStart
        self.NextCh()
        if self.ch == '*':
            self.NextCh()
            while True:
                if self.ch == '*':
                    self.NextCh()
                    if self.ch == '/':
                        level -= 1
                        if level == 0:
                            self.oldEols = self.line - line0
                            self.NextCh()
                            return True
                        self.NextCh()
                elif self.ch == Buffer.EOF:
                    return False
                else:
                    self.NextCh()
        else:
            if self.ch == Scanner.EOL:
                self.line -= 1
                self.lineStart = lineStart0
            self.pos = self.pos - 2
            self.buffer.setPos(self.pos + 1)
            self.NextCh()
        return False

    def CheckLiteral(self):
        lit = self.t.val
        if lit == "U":
            self.t.kind = 2
        elif lit == "F":
            self.t.kind = 3
        elif lit == "G":
            self.t.kind = 4
        elif lit == "X":
            self.t.kind = 5
        elif lit == "or":
            self.t.kind = 7
        elif lit == "and":
            self.t.kind = 8
        elif lit == "not":
            self.t.kind = 9
        elif lit == "true":
            self.t.kind = 10
        elif lit == "false":
            self.t.kind = 11

    def NextToken(self):
        while ord(self.ch) in self.ignore:
            self.NextCh()
        if (self.ch == '/' and self.Comment0()):
            return self.NextToken()

        self.t = Token()
        self.t.pos = self.pos
        self.t.col = self.pos - self.lineStart + 1
        self.t.line = self.line
        state = self.start[ord(self.ch)]
        buf = u''
        buf += self.ch
        self.NextCh()

        done = False
        while not done:
            if state == -1:
                self.t.kind = Scanner.eofSym  # NextCh already done
                done = True
            elif state == 0:
                self.t.kind = Scanner.noSym  # NextCh already done
                done = True
            elif state == 1:
                if (self.ch >= 'A' and self.ch <= 'Z'
                        or self.ch >= 'a' and self.ch <= 'z'):
                    buf += self.ch
                    self.NextCh()
                    state = 1
                elif self.ch == '_':
                    buf += self.ch
                    self.NextCh()
                    state = 2
                else:
                    self.t.kind = 1
                    self.t.val = buf
                    self.CheckLiteral()
                    return self.t
            elif state == 2:
                if (self.ch >= 'A' and self.ch <= 'Z'
                        or self.ch >= 'a' and self.ch <= 'z'):
                    buf += self.ch
                    self.NextCh()
                    state = 1
                else:
                    self.t.kind = Scanner.noSym
                    done = True
            elif state == 3:
                if self.ch == '>':
                    buf += self.ch
                    self.NextCh()
                    state = 4
                else:
                    self.t.kind = Scanner.noSym
                    done = True
            elif state == 4:
                self.t.kind = 6
                done = True
            elif state == 5:
                self.t.kind = 12
                done = True
            elif state == 6:
                self.t.kind = 13
                done = True

        self.t.val = buf
        return self.t

    def Scan(self):
        self.t = self.t.next
        self.pt = self.t.next
        return self.t

    def Peek(self):
        self.pt = self.pt.next
        while self.pt.kind > self.maxT:
            self.pt = self.pt.next

        return self.pt

    def ResetPeek(self):
        self.pt = self.t
