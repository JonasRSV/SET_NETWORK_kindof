
### Ever had this idea and had no idea if it would work?

This that idea, trying to make some kind of dynamic graphlike neural network.

> Example of Graph at start
![https://imgur.com/a/e7GdHOY](Image1)

> Example After 1 training Epoch
![https://imgur.com/a/96QsxJR](Image2)


The dynamic part comes from it changeing connections according to some rule.
The number of nodes is static. 


By using Tensorflow Eager Execution and a "multigraph" layout i was unable to make 
it adapt to data in any reasonable way!


Will try to rewrite it into a DAG or perhaps use some other propagation method than the one
currently in use... maybe some BFS or DFS thingy.


The problem with the current implementation is that the gradients are not guaranteed to be dependant on
input since there is no guarantee that input reaches output. Maybe that's not a problem... maybe 
output can be dependant only on state.. perhaps thats.. memory? Either way.. it does not work at the moment.


