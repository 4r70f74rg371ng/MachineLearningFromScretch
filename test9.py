tlist = ["<tr>",
"<title>",
"<ruby>",
"<template>",
"<table>",
"<td>",
"<col>",
"<em>",
"<th>",
"<DD>",
"</tr>",
"</td>",
"</table>"]

def EnumLenInt(tlist, ilen, now, result):
	if ilen == now:
		print "".join([tlist[r] for r in result])
	else:
		for i in range(0, len(tlist)):
			result[now] = i
			EnumLenInt(tlist, ilen, now+1, result)

# tlist * len
def EnumLen(tlist, ilen):
	EnumLenInt(tlist, ilen, 0, [0 for i in range(0, ilen)])

EnumLen(tlist, 10)

