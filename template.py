human_chat = '''
<div class="human">
    <div class="right">
        <h3>Me</h3>
        <p>{{MSG}}</p>
    <div/>
</div>
'''

ai_chat = '''
<div class="ai">
    <div class="left">
        <h3>AI</h3>
        <p>{{MSG}}</p>
    </div>
</div>
'''

css = '''
<style>
.human{
    background-color: #3C3C3C;
    border: 1px solid black;
    border-radius: 25px;
    margin: 10px 0px;
    align-self: right;
}

.ai{
    background-color: #27302C;
    border: 1px solid black;
    border-radius: 25px
}


.bot--img{
    max-height : 20px;
}

.right{
    text-align: right;
    padding-right: 20px;
}

.left{
    text-align: left;
    padding-left: 20px;
}
'''