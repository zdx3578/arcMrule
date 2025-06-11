%% ------------------------------------------------
%% bias.pl – mode declarations & metarules
%% ------------------------------------------------
:- modeh(1, transform(+id, -color)).    %% hypothesis head
%% available body predicates
:- modeb(*, holes(+id, #int)).
:- modeb(*, color(+id, #color)).
:- modeb(1, greater_than(#int,#int)).   %% helper
:- modeb(1, equal(#int,#int)).
%% determinations
:- determination(transform/2, holes/2).
:- determination(transform/2, color/2).
:- determination(transform/2, greater_than/2).
:- determination(transform/2, equal/2).

%% Metarules
metarule(identity,[P,Q],
         ([P,A,B] :- [[Q,A,B]])).

metarule(chain,[P,Q,R],
         ([P,A,B] :- [[Q,A,C],[R,C,B]])).

%% max program size / body length can be tuned
max_body(3).