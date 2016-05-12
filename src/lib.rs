// http://semantic-domain.blogspot.co.uk/2013/11/antimirov-derivatives-for-regular.html

#[macro_use]
extern crate maplit;
use std::collections::{BTreeMap,BTreeSet, VecDeque};
use std::rc::Rc;
use std::ops::BitOr;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Re {
    Byte(char),
    Nil,
    Bot,
    Seq (RcRe, RcRe),
    Alt (RcRe, RcRe),
    Star (RcRe),
}

/*
let rec null = function
  | C _ | Bot -> false
  | Nil | Star _ -> true
  | Alt(r1, r2) -> null r1 || null r2
  | Seq(r1, r2) -> null r1 && null r2

   */

/*
    module R = Set.Make(struct type t = re let compare = compare end)
    let rmap f rs = R.fold (fun r -> R.add (f r)) rs R.empty

    module M = Map.Make(R)
    module I = Set.Make(struct type t = int let compare = compare end)
*/

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RcRe(Rc<Re>);
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Res(BTreeSet<RcRe>);
// type M<Y> = BTreeMap<R, Y>;

impl Res {
    fn new() -> Res {
        Res(BTreeSet::new())
    }

    fn rmap<F: Fn(&RcRe) -> RcRe>(&self, f: F) -> Res {
        Res(self.0.iter().map(f).collect())
    }

    fn deriv(&self, c: char) -> Res {
        Res(self.0.iter().flat_map(|r| r.deriv(c).0.into_iter()).collect())
    }

    fn is_null(&self) -> bool {
        self.0.iter().any(|r| r.is_null())
    }
}

impl<'a> BitOr for &'a Res {
    type Output = Res;
    fn bitor(self, other:Self) -> Res {
        Res(&self.0 | &other.0)
    }
}

impl From<BTreeSet<RcRe>> for Res {
    fn from(s: BTreeSet<RcRe>) -> Res {
        Res(s)
    }
}

impl RcRe {
    fn is_null(&self) -> bool {
        match &*self.0 {
            &Re::Byte(_) | &Re::Bot => false,
            &Re::Nil | &Re::Star(_) => true,
            &Re::Alt(ref l, ref r) => l.is_null() || r.is_null(),
            &Re::Seq(ref l, ref r) => l.is_null() && r.is_null(),
        }
    }
    pub fn nil() -> RcRe {
        RcRe(Rc::new(Re::Nil))
    }
    pub fn lit(c: char) -> RcRe {
        RcRe(Rc::new(Re::Byte(c)))
    }
    pub fn seq(l: RcRe, r: RcRe) -> RcRe {
        RcRe(Rc::new(Re::Seq(l, r)))
    }
    pub fn alt(l: RcRe, r: RcRe) -> RcRe {
        RcRe(Rc::new(Re::Alt(l, r)))
    }
    pub fn star(r: RcRe) -> RcRe {
        RcRe(Rc::new(Re::Star(r)))
    }

    pub fn deriv(&self, c:char) -> Res {
        match &*self.0 {
            &Re::Byte(m) if c == m => Res(vec![RcRe::nil()].into_iter().collect()),
            &Re::Byte(_) | &Re::Nil | &Re::Bot => Res::new(),
            &Re::Alt(ref l, ref r) => &l.deriv(c) | &r.deriv(c),
            &Re::Seq(ref l, ref r) =>
                &l.deriv(c).rmap(|l2| RcRe::seq(l2.clone(), r.clone()))
                | &if l.is_null() { r.deriv(c) } else { Res::new() },
            &Re::Star(ref r) => r.deriv(c).rmap(|r2| RcRe::seq(r2.clone(), RcRe::star(r.clone()))),
        }
    }

    /// From Antimirov:
    ///
    ///
    pub fn lf(&self) -> BTreeSet<(char, RcRe)> {
//         println!("lf: {:x} <- {:?}; null? {:?}", hash(&self), self, self.is_null());
        let res = match &*self.0 {
            &Re::Nil | &Re::Bot => btreeset!{},
            &Re::Byte(m) => btreeset!{(m, RcRe::nil())},
            &Re::Alt(ref l, ref r) => &l.lf() | &r.lf(),
            &Re::Star(ref r) => Self::prod(r.lf(), self.clone()),
            &Re::Seq(ref l, ref r) => Self::prod(l.lf(), r.clone())
                .union(&if !l.is_null() { btreeset!{} } else { r.lf() })
                .cloned()
                .collect()
        };
//         println!("lf: {:x} {:?} -> {:#?}", hash(&self), self, res);
        res
    }

    fn prod(l: BTreeSet<(char, RcRe)>, t: RcRe) -> BTreeSet<(char, RcRe)> {
        let res = match &*t.0 {
            &Re::Nil => btreeset!{},
            &Re::Bot => l.clone(),
            _ => l.clone().into_iter()
                    .map(|(x, ref p)| (x, if p == &Self::nil() { t.clone() } else { Self::seq(p.clone(), t.clone()) })).collect()
        };
//         println!("prod: {:?} {:?} -> {:#?}", l, t, res);
        res
    }


    /// From Antimirov:
    ///
    /// (PD[0], Δ[0], τ[0]) = (∅, {t}, ∅)
    ///
    /// PD[i+1] := PD[i] ∪  Δ[i]
    ///
    /// Δ[i+1] := ⋃(p∈ Δ[i]) {q | (x, q) ∈ lf(p)∧q ∉ PD[i+1]}
    /// τ[i+1] := τ[i]∪  {(p, x, q) | p ∈ Δ[i] ∧ (x, q) ∊ lf(p)}
    ///
    pub fn make_nfa(&self) -> NFA {
        #[derive(Debug, Default, PartialEq, Eq, Hash)]
        struct State {
            pd: BTreeSet<RcRe>,
            delta: BTreeSet<RcRe>,
            tau: BTreeSet<(RcRe, char, RcRe)>,
        }
        let mut state : State = Default::default();
        state.delta.insert(self.clone());

        loop {
            let mut next : State = Default::default();
//             println!("S: {:#?}", state);
            next.pd.extend(state.pd.iter().cloned());
            next.pd.extend(state.delta.iter().cloned());

            next.delta = state.delta.iter()
                .flat_map(|p| p.lf().into_iter().map(|(x, q)| q))
                .filter(|q| !next.pd.contains(q))
                .collect();

            next.tau = state.tau.iter().cloned().chain(
                    state.delta.iter()
                    .flat_map(|p| p.lf().into_iter().map(move |(x, q)| (p.clone(), x, q))))
                    .collect();

//             println!("N@{:16x}: pd: {:?}; delta: {:?}; tau: {:?}",
//                  hash(&next), next.pd.len(), next.delta.len(), next.tau.len());

            if state == next { break; }
            state = next;
        }

        // let idx = state.pd.iter().enumerate().map(|(i, r)| (r.clone(), i)).collect::<BTreeMap<_, _>>();

        // let idx = state.pd.iter().enumerate().map(|(i, r)| (r.clone(), i)).collect::<BTreeMap<_, _>>();
        let initial = 0;
        let mut transitions = btreemap!{};
//         println!("digraph g{{");
//         println!("start -> N{};", initial);

        let mut idx = btreemap!{self.clone() => initial};

        let mut counter = 0;

        for (p, x, q) in state.tau {
//             println!("N{} -> N{} [label=\"{}\"];", idx[&p], idx[&q], x);
            let pi = *idx.entry(p.clone()).or_insert_with(|| { counter += 1; counter });
            let qi = *idx.entry(q.clone()).or_insert_with(|| { counter += 1; counter });

            let ent = transitions.entry((pi, x)).or_insert_with(|| vec!{});
            if !ent.contains(&qi) {
                ent.push(qi);
            }
        }
        for state in state.pd.iter() {
//             println!("N{}[label=\"{:?}\"];", idx[state], state);
        }
//         println!("}}");

        let finals = state.pd.iter().filter(|p| p.is_null()).map(|p| idx[p]).collect();

        NFA { transition: transitions, initial: initial, finals: finals }
    }
}

fn hash<T : ::std::hash::Hash>(x: T) -> u64 {
    use std::hash::*;
    let mut h = SipHasher::new();
    x.hash(&mut h);
    h.finish()
}

impl fmt::Debug for RcRe {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "{:?}", *self.0)
    }
}

impl std::ops::Mul for RcRe {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        RcRe::seq(self, other)
    }
}

impl std::ops::Add for RcRe {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        RcRe::alt(self, other)
    }
}


#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct NFA {
    initial: usize,
    transition: BTreeMap<(usize, char), Vec<usize>>,
    finals: BTreeSet<usize>,
}

impl NFA {
    pub fn matches(&self, s: &str) -> bool {
//         // println!("Matching: {:?} against {:?}", s, self);
        let mut states = vec![self.initial];
        let mut next = vec![];
        for c in s.chars() {
            next.clear();
            // print!("{:?} @ {:?}", states, c);
            next.extend(states.iter().cloned()
                .flat_map(|state| self.transition.get(&(state, c))
                        .into_iter()
                        .flat_map(|states| states.into_iter()))
                .cloned());
            // println!(" -> {:?}", next);
            std::mem::swap(&mut states, &mut next);
        }

        states.into_iter().any(|s| self.finals.contains(&s))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
struct DFA {
    size: usize,
    fail: usize,
    trans: Vec<(usize, char, usize)>,
    finalp: BTreeSet<usize>,
}

/*
fn charfold<F: Fn(V, char) -> V, V>(f: F, zero: V) -> V {
    (0char..256).fold(zero, f)
}

impl DFA{
    fn build(re: RcRe) -> DFA {
        let mut dfa : DFA = Default::default();
        let mut states = BTreeMap::new();
        let mut queue = VecDeque::new();

        queue.push_back(re);

        while let Some(exp) = queue.pop_front() {
            // let exp = exp.normalize();
            // LookupOrInsert
            let idx = states.len();
            let idx = *states.entry(exp.clone()).or_insert(idx);
            if exp.is_null() {
                dfa.finalp.insert(idx);
            }
        }

        dfa
    }
}
*/


#[cfg(test)]
mod tests {
    use super::{RcRe,Res};
    #[test]
    fn it_works() {
        let re = RcRe::star(RcRe::lit('a'));
//         println!("{:?} -> {:?}", re, re.is_null());
        assert!(re.is_null());
        let deriv1 = re.deriv('a');
//         println!("{:?} / {:?} -> {:?}; {:?}", re, "a", deriv1, deriv1.is_null());
        assert!(deriv1.is_null());
        let deriv2 = deriv1.deriv('a');
//         println!("{:?} / {:?} -> {:?}; {:?}", re, "aa", deriv2, deriv2.is_null());
        assert!(deriv2.is_null());
    }

    #[test]
    fn should_build_nfa() {
        use super::RcRe as R;
        let re = R::lit('a') * R::lit('b') * R::lit('a') * R::lit('b') +
                R::lit('a') * R::lit('b') * R::lit('b') * R::lit('b');
//         println!("Re: {:?}", re);
        let nfa = re.make_nfa();
//         println!("NFA: {:?}", nfa);
//         println!("---");
        assert!(nfa.matches("abbb"));
//         println!("---");
        assert!(nfa.matches("abab"));
//         println!("---");
        assert!(!nfa.matches("abaa"));
        assert!(!nfa.matches("aba"));
    }

    #[test]
    fn should_build_nfa2() {
        use super::RcRe as R;
        let re = R::star(R::lit('a') * R::lit('b')) * R::lit('c');
//         println!("Re: {:?}", re);
        let nfa = re.make_nfa();
//         println!("NFA: {:?}", nfa);
//         println!("---");
        assert!(!nfa.matches("aabbc"));
        assert!(nfa.matches("abc"));
//         println!("---");
        assert!(nfa.matches("c"));
//         println!("---");
        assert!(nfa.matches("ababc"));
        assert!(!nfa.matches("aba"));
    }
    #[test]
    fn should_build_nfa3() {
        use super::RcRe as R;
        let re = R::lit('a') * R::lit('b') * R::lit('c') + R::lit('e') * R::lit('d') * R::lit('c') ;
//         println!("Re: {:?}", re);
        let nfa = re.make_nfa();
//         println!("NFA: {:?}", nfa);
//         println!("---");
        assert!(nfa.matches("abc"));
        assert!(nfa.matches("edc"));
        assert!(!nfa.matches("bc"));
        assert!(!nfa.matches("dc"));
        assert!(!nfa.matches("c"));
    }
    #[test]
    fn should_build_nfa4() {
        use super::RcRe as R;
        let re = R::lit('a') * R::lit('e');
//         println!("Re: {:?}", re);
        let nfa = re.make_nfa();
//         println!("NFA: {:?}", nfa);
//         println!("---");
        assert!(nfa.matches("ae"));
        assert!(!nfa.matches("e"));
        assert!(!nfa.matches("ea"));

        // assert!(false);
    }
}
