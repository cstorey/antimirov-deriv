#![feature(test)]
extern crate test;
extern crate antimirov;
extern crate regex;

use antimirov::{NFA, RcRe as R};
fn test_re() -> R {
    R::lit('a') * R::lit('b') * R::lit('a') * R::lit('b') +
    R::lit('a') * R::lit('b') * R::lit('b') * R::lit('b')
}

fn bench_nfa<F: Fn(&NFA, &str) -> bool>(b: &mut test::Bencher,
        re: R, s: &str, matchp: bool, f: F) {
    let nfa = NFA::build(&re);
    let arg = "abba";
    b.bytes = arg.len() as u64;
    assert_eq!(f(&nfa, s), matchp);
    b.iter(|| f(&nfa, s))

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
