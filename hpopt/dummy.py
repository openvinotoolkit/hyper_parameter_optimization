# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
import logging

from hpopt.base import HpOpt
import hpopt

logger = logging.getLogger(__name__)


class DummyOpt(HpOpt):
    def __init__(self,
                 **kwargs):
        super(DummyOpt, self).__init__(**kwargs)

        hpo_file_path = hpopt.get_status_path(self.save_path)

        if self.resume is True and os.path.exists(hpo_file_path):
            with open(hpo_file_path, 'rt') as json_file:
                self.hpo_status = json.load(json_file)
                json_file.close()

            if self.hpo_status['search_space'] != {ss: self.search_space[ss].__dict__ for ss in self.search_space}:
                logger.error("Search space is changed. Stop resuming.")
                raise ValueError("Search space is changed.")
        else:
            self.hpo_status['config_list'] = []

    def get_next_sample(self):
        return None
