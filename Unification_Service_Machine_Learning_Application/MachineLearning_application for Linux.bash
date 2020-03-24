
#!/bin/bash
args=("$@")

function goto
{
label=$1
cmd=$(sed -n "/$label:/{:a;n;p;ba};" $0 | grep -v ':$')
eval "$cmd"
exit
}

$cd\Python37\python Machine_Learning_Tool\Machine_Learning_Tool_Unification_Service.py