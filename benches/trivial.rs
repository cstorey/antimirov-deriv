#![feature(test)]
extern crate test;
extern crate antimirov;
extern crate regex;

mod antimorov {
    use antimirov::RcRe as R;
    use test;
    #[bench]
    fn non_matching(b: &mut test::Bencher) {
        let re = R::lit('a') * R::lit('b') * R::lit('a') * R::lit('b') +
                 R::lit('a') * R::lit('b') * R::lit('b') * R::lit('b');

        let nfa = re.make_nfa();
        let arg = "abba";
        b.bytes = arg.len() as u64;
        assert!(!nfa.matches(&arg));
        b.iter(|| nfa.matches(&arg))
    }

    #[bench]
    fn matching_ok(b: &mut test::Bencher) {
        let re = R::lit('a') * R::lit('b') * R::lit('a') * R::lit('b') +
                 R::lit('a') * R::lit('b') * R::lit('b') * R::lit('b');

        let nfa = re.make_nfa();
        let arg = "abab";
        b.bytes = arg.len() as u64;
        assert!(nfa.matches(&arg));
        b.iter(|| nfa.matches(&arg))
    }
}

mod antimorov_rec {
    use antimirov::RcRe as R;
    use test;
    #[bench]
    fn non_matching(b: &mut test::Bencher) {
        let re = R::lit('a') * R::lit('b') * R::lit('a') * R::lit('b') +
                 R::lit('a') * R::lit('b') * R::lit('b') * R::lit('b');

        let nfa = re.make_nfa();
        let arg = "abba";
        b.bytes = arg.len() as u64;
        assert!(!nfa.matches_rec(&arg));
        b.iter(|| nfa.matches_rec(&arg))
    }

    #[bench]
    fn matching_ok(b: &mut test::Bencher) {
        let re = R::lit('a') * R::lit('b') * R::lit('a') * R::lit('b') +
                 R::lit('a') * R::lit('b') * R::lit('b') * R::lit('b');

        let nfa = re.make_nfa();
        let arg = "abab";
        b.bytes = arg.len() as u64;
        assert!(nfa.matches_rec(&arg));
        b.iter(|| nfa.matches_rec(&arg))
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
