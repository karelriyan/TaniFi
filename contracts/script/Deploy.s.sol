// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Script} from "forge-std/Script.sol";
import {TaniVault} from "../src/TaniVault.sol";

/// @notice Foundry script that deploys TaniVault using PRIVATE_KEY from env.
contract Deploy is Script {
    function run() external {
        uint256 deployerKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerKey);

        new TaniVault();

        vm.stopBroadcast();
    }
}
