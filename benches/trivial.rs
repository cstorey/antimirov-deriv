#![feature(test)]
extern crate test;
extern crate antimirov;
extern crate regex;

use std::iter;

use antimirov::{NFA, RcRe as R};
fn test_re() -> R {
    R::lit('a') * R::lit('b') * R::lit('a') * R::lit('b') +
    R::lit('a') * R::lit('b') * R::lit('b') * R::lit('b')
}

fn pathological(n: usize) -> (R, String) {
    // a?^na^n matches a^n:
    let a = R::lit('a');
    let a_maybe = a.clone() + R::nil();

    let re =
        iter::repeat(a_maybe).take(n).fold(R::nil(), |a, b| a * b) *
        iter::repeat(a).take(n).fold(R::nil(), |a, b| a * b);

    (re, iter::repeat('a').take(n).collect())
}

fn bench_nfa<F: Fn(&NFA, &str) -> bool>(b: &mut test::Bencher,
        re: R, s: &str, matchp: bool, f: F) {
    let nfa = NFA::build(&re);
    let arg = "abba";
    b.bytes = arg.len() as u64;
    assert_eq!(f(&nfa, s), matchp);
    // println!("NFA: {:?}", nfa);
    b.iter(|| f(&nfa, s))

}

fn bench_pathological(b: &mut test::Bencher, n: usize, f: fn(&NFA, &str) -> bool) {
    let (re, s) = pathological(n);
    bench_nfa(b, re, &s, true, |nfa, arg| f(&nfa, &arg))
}

mod antimorov {
    use antimirov::{NFA, RcRe as R};
    use test;
    #[bench]
    fn non_matching(b: &mut test::Bencher) {
        super::bench_nfa(b, super::test_re(), "abba", false,
                |nfa, arg| nfa.matches(&arg))
    }

    #[bench]
    fn matching_ok(b: &mut test::Bencher) {
        super::bench_nfa(b, super::test_re(), "abab", true,
                |nfa, arg| nfa.matches(&arg))
    }

    #[bench] fn pathological_01(b: &mut test::Bencher) { super::bench_pathological(b, 1, NFA::matches); }
    #[bench] fn pathological_02(b: &mut test::Bencher) { super::bench_pathological(b, 2, NFA::matches); }
    #[bench] fn pathological_04(b: &mut test::Bencher) { super::bench_pathological(b, 4, NFA::matches); }
    #[bench] fn pathological_08(b: &mut test::Bencher) { super::bench_pathological(b, 8, NFA::matches); }
    #[bench] fn pathological_16(b: &mut test::Bencher) { super::bench_pathological(b, 16, NFA::matches); }
    #[bench] fn pathological_32(b: &mut test::Bencher) { super::bench_pathological(b, 32, NFA::matches); }
}

mod antimorov_rec {
    use antimirov::{NFA, RcRe as R};
    use test;
    #[bench]
    fn non_matching(b: &mut test::Bencher) {
        super::bench_nfa(b, super::test_re(), "abba", false,
                |nfa, arg| nfa.matches_rec(&arg))
    }

    #[bench]
    fn matching_ok(b: &mut test::Bencher) {
        super::bench_nfa(b, super::test_re(), "abab", true,
                |nfa, arg| nfa.matches_rec(&arg))
    }

    #[bench] fn pathological_01(b: &mut test::Bencher) { super::bench_pathological(b, 1, NFA::matches_rec); }
    #[bench] fn pathological_02(b: &mut test::Bencher) { super::bench_pathological(b, 2, NFA::matches_rec); }
    #[bench] fn pathological_04(b: &mut test::Bencher) { super::bench_pathological(b, 4, NFA::matches_rec); }
    #[bench] fn pathological_08(b: &mut test::Bencher) { super::bench_pathological(b, 8, NFA::matches_rec); }
    #[bench] fn pathological_16(b: &mut test::Bencher) { super::bench_pathological(b, 16, NFA::matches_rec); }
    #[bench] fn pathological_32(b: &mut test::Bencher) { super::bench_pathological(b, 32, NFA::matches_rec); }
}

mod antimorov_bt {
    use antimirov::{NFA, RcRe as R};
    use test;
    #[bench]
    fn non_matching(b: &mut test::Bencher) {
        super::bench_nfa(b, super::test_re(), "abba", false,
                |nfa, arg| nfa.matches_bt(&arg))
    }

    #[bench]
    fn matching_ok(b: &mut test::Bencher) {
        super::bench_nfa(b, super::test_re(), "abab", true,
                |nfa, arg| nfa.matches_bt(&arg))
    }

    #[bench] fn pathological_01(b: &mut test::Bencher) { super::bench_pathological(b, 1, NFA::matches_bt); }
    #[bench] fn pathological_02(b: &mut test::Bencher) { super::bench_pathological(b, 2, NFA::matches_bt); }
    #[bench] fn pathological_04(b: &mut test::Bencher) { super::bench_pathological(b, 4, NFA::matches_bt); }
    #[bench] fn pathological_08(b: &mut test::Bencher) { super::bench_pathological(b, 8, NFA::matches_bt); }
    #[bench] fn pathological_16(b: &mut test::Bencher) { super::bench_pathological(b, 16, NFA::matches_bt); }
    #[bench] fn pathological_32(b: &mut test::Bencher) { super::bench_pathological(b, 32, NFA::matches_bt); }
}


mod rust_regex {
    use regex::Regex;
    use test;
    #[bench]
    fn non_matching(b: &mut test::Bencher) {
        let re = Regex::new(r"^(abab|abbb)$").unwrap();

        let arg = "abba";
        b.bytes = arg.len() as u64;
        assert!(!re.is_match(&arg));
        b.iter(|| re.is_match(&arg))
    }

    #[bench]
    fn matching_ok(b: &mut test::Bencher) {
        let re = Regex::new(r"^(abab|abbb)$").unwrap();

        let arg = "abab";
        b.bytes = arg.len() as u64;
        assert!(re.is_match(&arg));
        b.iter(|| re.is_match(&arg))
    }
}
