from smartcampus_ai.evaluation_data import OUT_OF_DOMAIN, X_TEST, Y_TEST


def test_held_out_set_is_balanced_and_disjoint_from_out_of_domain() -> None:
    assert len(X_TEST) == len(Y_TEST) == 15
    assert {label: Y_TEST.count(label) for label in set(Y_TEST)} == {
        "inscriptions": 5,
        "attestations": 5,
        "paiements": 5,
    }
    assert set(X_TEST).isdisjoint(OUT_OF_DOMAIN)
