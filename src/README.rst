README
======



Some icons by `Yusuke Kamiyamane <https://p.yusukekamiyamane.com/>`_.
Licensed under a `Creative Commons Attribution 3.0 License <https://creativecommons.org/licenses/by/3.0/>`_.

Simulation Example
==================

Network
-------
* N = 3
* D = 1
* reset_firing_times_ptr_threshold = rftth = 6

* n_fired_total = 0 ( .. += n_fired_0)
* n_fired_total_m1 = 0
* n_fired_m1_to_end = 0
* n_fired = 0 ( .. += n_fired_0)
* n_fired_0 = 0 ( .. = firing_counts[firing_counts_idx])
* n_fired_m1 = 0 ( .. = firing_counts[firing_counts_idx_m1])
* (pointer) firing_times_write = firing_times ( .. += n_fired_0)
* (pointer) firing_times_read = firing_times ( .. += n_fired_m1)
* (pointer) firing_idcs_write = firing_idcs ( .. += n_fired_0)
* (pointer) firing_idcs_read = firing_idcs ( .. += n_fired_m1)
* firing_counts_idx = 1
* firing_counts_idx_m1 = 1
* firing_counts_idx_end = 1

t = 0
-----

.. list-table::
    :stub-columns: 1

    * - Fired
      - 0
      - 0
      - 0

.. list-table::
    :stub-columns: 1

    * - firing_counts
      - 0
      - 0*
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0

n_fired_0 = 0

.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * -
      -
      - t>=D
      - n_fired
      - n_fired _total
      - n_fired _total_m1
      - firing _counts _idx
      - firing _counts _idx _m1
      - n_fired_total > n_fired_total _m1
      - n_fired _m1 _to_end
      - firing_times _write
      - firing_idcs _write
      - firing_counts _write
      - firing _idcs _read
      - firing _counts _read
    * -
      -
      - False
      - 0 (+0)
      - 0 (+0)
      - 0
      - 3 (+2)
      - 1
      - False
      - 0
      -
      -
      -
      -
      -
    * - n_fired_total > rftth
      - False
      -
      -
      -
      -
      -
      -
      -
      -
      - firing_times (+0)
      - firing_idcs (+0)
      - firing_counts + 2 (+2)
      -
      -
    * - n_fired_total _m1 > rftth
      - False
      -
      -
      -
      -
      -
      -
      -
      -
      -
      -
      -
      - firing_times (+0)
      - firing_idcs (+0)


t = 1
-----

.. list-table::
    :stub-columns: 1

    * - Fired
      - 0
      - 1
      - 1

.. list-table::
    :stub-columns: 1

    * - firing_counts
      - 0
      - 0
      - 0
      - 2*
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0

n_fired_0 = 2

.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * -
      -
      - t>=D
      - n_fired
      - n_fired _total
      - n_fired _total_m1
      - firing _counts _idx
      - firing _counts _idx _m1
      - n_fired_total > n_fired_total _m1
      - n_fired _m1 _to_end
      - firing_times _write
      - firing_idcs _write
      - firing_counts _write
      - firing_idcs _read
      - firing_counts _read
    * -
      -
      - False
      - 2 (0+2)
      - 2 (0+2)
      - 0
      - 5 (+2)
      - 1
      - True
      - 2 (0+2)
      -
      -
      -
      -
      -
    * - n_fired_total > rftth
      - False
      -
      -
      -
      -
      -
      -
      -
      -
      - firing_times + 2 (+2)
      - firing_idcs + 2 (+2)
      - firing_counts + 4 (+2)
      -
      -
    * - n_fired_total _m1 > rftth
      - False
      -
      -
      -
      -
      -
      -
      -
      -
      -
      -
      -
      - firing_times
      - firing_idcs


t = 2
-----

.. list-table::
    :stub-columns: 1

    * - Fired
      - 1
      - 1
      - 1

.. list-table::
    :stub-columns: 1

    * - firing_counts
      - 0
      - 0
      - 0
      - 2
      - 0
      - 3*
      - 0
      - 0
      - 0
      - 0
      - 0
      - 0

* n_fired_0 = 3
* n_fired_m1 = firing_counts[firing_counts_idx_m1] = 0


.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * -
      -
      - t>=D
      - n_fired
      - n_fired _total
      - n_fired _total_m1
      - firing _counts _idx
      - firing _counts _idx _m1
      - n_fired_total > n_fired_total _m1
      - n_fired _m1 _to_end
      - firing_times _write
      - firing_idcs _write
      - firing_counts _write
      - firing_idcs _read
      - firing_counts _read
    * -
      -
      - True
      - 5 (2+3)
      - 5 (2+3-0)
      - 0 (+0)
      - 7 (+2)
      - 3 (+2)
      - True
      - 5 (2+3-0)
      -
      -
      -
      -
      -
    * - n_fired_total > rftth
      - False
      -
      -
      -
      -
      -
      -
      -
      -
      - firing_times + 5 (+3)
      - firing_idcs + 5 (+3)
      - firing_counts + 6 (+2)
      -
      -
    * - n_fired_total _m1 > rftth
      - False
      -
      -
      -
      -
      -
      -
      -
      -
      -
      -
      -
      - firing_times
      - firing_idcs


t = 3
-----

.. list-table::
    :stub-columns: 1

    * - Fired
      - 1
      - 0
      - 0

.. list-table::
    :stub-columns: 1

    * - firing_counts
      - 0
      - 0
      - 0
      - 2
      - 0
      - 3
      - 0
      - 1*
      - 0
      - 0
      - 0
      - 0

* n_fired_0 = 1
* n_fired_m1 = 2

.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * -
      -
      - t>=D
      - n_fired
      - n_fired _total
      - n_fired _total_m1
      - firing _counts _idx
      - firing _counts _idx _m1
      - n_fired_total > n_fired_total _m1
      - n_fired _m1 _to_end
      - firing_times _write
      - firing_idcs _write
      - firing_counts _write
      - firing_idcs _read
      - firing_counts _read
    * -
      -
      - True
      - 4 (5+1-2)
      - 6 (5+1)
      - 2 (0+2)
      - 9 (+2)
      - 5 (+2)
      - True
      - 4 (5+1-2)
      -
      -
      -
      -
      -
    * - n_fired_total > rftth
      - False
      -
      -
      -
      -
      -
      -
      -
      -
      - firing_times + 6 (+1)
      - firing_idcs + 6 (+1)
      - firing_counts + 8 (+2)
      -
      -
    * - n_fired_total _m1 > rftth
      - False
      -
      -
      -
      -
      -
      -
      -
      -
      -
      -
      -
      - firing_times + 2 (+2)
      - firing_idcs + 2 (+2)


t = 4
-----

.. list-table::
    :stub-columns: 1

    * - Fired
      - 1
      - 1
      - 1

.. list-table::
    :stub-columns: 1

    * - firing_counts
      - 0
      - 0
      - 0
      - 2
      - 0
      - 3
      - 0
      - 1
      - 0
      - 3*
      - 0
      - 0

* n_fired_0 = 1
* n_fired_m1 = 3

* (n_fired_total_m1 > 6) = False
    * (pointer) firing_times_read = firing_times + 5 (+3)
    * (pointer) firing_idcs_read = firing_idcs + 5 (+3)

.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * -
      -
      - t>=D
      - n_fired
      - n_fired _total
      - n_fired _total_m1
      - firing _counts _idx
      - firing _counts _idx _m1
      - n_fired_total > n_fired_total _m1
      - n_fired _m1 _to_end
      - firing_times _write
      - firing_idcs _write
      - firing_counts _write
      - firing_idcs _read
      - firing_counts _read
    * -
      -
      - True
      - 4 (4+3-3)
      - 9 (6+3)
      - 5 (2+3)
      - 11 (+2)
      - 7 (+2)
      - True
      - 4 (4+3-3)
      -
      -
      -
      -
      -
    * - n_fired_total > rftth
      - True
      -
      -
      - 0
      -
      - 1
      -
      -
      -
      - firing_times
      - firing_idcs
      - firing_counts
      -
      -
    * - n_fired_total _m1 > rftth
      - False
      -
      -
      -
      -
      -
      -
      -
      -
      -
      -
      -
      - firing_times + 5 (+3)
      - firing_idcs + 5 (+3)

t = 5
-----

.. list-table::
    :stub-columns: 1

    * - Fired
      - 1
      - 1
      - 0

.. list-table::
    :stub-columns: 1

    * - firing_counts
      - 0
      - 2*
      - 0
      - 2
      - 0
      - 3
      - 0
      - 1
      - 0
      - 3
      - 0
      - 0

* n_fired_0 = 2
* n_fired_m1 = 1


.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * -
      -
      - t>=D
      - n_fired
      - n_fired _total
      - n_fired _total_m1
      - firing _counts _idx
      - firing _counts _idx _m1
      - n_fired_total > n_fired_total _m1
      - n_fired _m1 _to_end
      - firing_times _write
      - firing_idcs _write
      - firing_counts _write
      - firing_idcs _read
      - firing_counts _read
    * -
      -
      - True
      - 5 (4+2-1)
      - 2 (0+2)
      - 6 (5+1)
      - 3 (+2)
      - 9 (+2)
      - False
      - 3 (4-1)
      -
      -
      -
      -
      -
    * - n_fired_total > rftth
      - False
      -
      -
      -
      -
      -
      -
      -
      -
      - firing_times + 2 (+2)
      - firing_idcs + 2 (+2)
      - firing_counts + 2 (+2)
      -
      -
    * - n_fired_total _m1 > rftth
      - False
      -
      -
      -
      -
      -
      -
      -
      -
      -
      -
      -
      - firing_times + 6 (+1)
      - firing_idcs + 6 (+1)


t = 6
-----

.. list-table::
    :stub-columns: 1

    * - Fired
      - 1
      - 0
      - 0

.. list-table::
    :stub-columns: 1

    * - firing_counts
      - 0
      - 2
      - 0
      - 1*
      - 0
      - 3
      - 0
      - 1
      - 0
      - 3
      - 0
      - 0

* n_fired_0 = 1
* n_fired_m1 = 2


.. list-table::
    :header-rows: 1
    :stub-columns: 1

    * -
      -
      - t>=D
      - n_fired
      - n_fired _total
      - n_fired _total_m1
      - firing _counts _idx
      - firing _counts _idx _m1
      - n_fired_total > n_fired_total _m1
      - n_fired _m1 _to_end
      - firing_times _write
      - firing_idcs _write
      - firing_counts _write
      - firing_idcs _read
      - firing_counts _read
    * -
      -
      - True
      - 3 (5+1-3)
      - 3 (2+1)
      - 9 (6+3)
      - 5 (+2)
      - 11 (+2)
      - False
      - 0 (3-3)
      -
      -
      -
      -
      -
    * - n_fired_total > rftth
      - False
      -
      -
      -
      -
      -
      -
      -
      -
      - firing_times + 3 (+1)
      - firing_idcs + 3 (+1)
      - firing_counts + 4 (+2)
      -
      -
    * - n_fired_total _m1 > rftth
      - True
      -
      -
      - 0
      -
      -
      - 1
      -
      -
      -
      -
      -
      - firing_times
      - firing_idcs