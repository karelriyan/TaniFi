// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console} from "forge-std/Test.sol";
import {MockIDRX} from "../src/MockIDRX.sol";

contract MockIDRXTest is Test {
    MockIDRX public idrx;

    address public owner = address(this);
    address public alice = address(0x1);
    address public bob = address(0x2);
    address public minter = address(0x3);

    function setUp() public {
        idrx = new MockIDRX();
    }

    // ============ Constructor Tests ============

    function test_Constructor() public view {
        assertEq(idrx.name(), "Mock IDRX Stablecoin");
        assertEq(idrx.symbol(), "IDRX");
        assertEq(idrx.decimals(), 2);
        assertEq(idrx.owner(), owner);
        assertTrue(idrx.minters(owner));
    }

    // ============ Decimals Tests ============

    function test_Decimals() public view {
        assertEq(idrx.decimals(), 2);
    }

    // ============ Minting Tests ============

    function test_MintAsOwner() public {
        uint256 amount = 1000 * 10 ** idrx.decimals();
        idrx.mint(alice, amount);
        assertEq(idrx.balanceOf(alice), amount);
    }

    function test_MintAsMinter() public {
        idrx.addMinter(minter);

        vm.prank(minter);
        uint256 amount = 1000 * 10 ** idrx.decimals();
        idrx.mint(alice, amount);

        assertEq(idrx.balanceOf(alice), amount);
    }

    function test_RevertMintNotMinter() public {
        vm.prank(alice);
        vm.expectRevert(MockIDRX.NotMinter.selector);
        idrx.mint(bob, 1000);
    }

    function test_RevertMintToZeroAddress() public {
        vm.expectRevert(MockIDRX.InvalidAddress.selector);
        idrx.mint(address(0), 1000);
    }

    // ============ Minter Management Tests ============

    function test_AddMinter() public {
        assertFalse(idrx.minters(minter));

        idrx.addMinter(minter);

        assertTrue(idrx.minters(minter));
    }

    function test_AddMinterEmitsEvent() public {
        vm.expectEmit(true, false, false, false);
        emit MockIDRX.MinterAdded(minter);

        idrx.addMinter(minter);
    }

    function test_RevertAddMinterNotOwner() public {
        vm.prank(alice);
        vm.expectRevert();
        idrx.addMinter(minter);
    }

    function test_RevertAddMinterZeroAddress() public {
        vm.expectRevert(MockIDRX.InvalidAddress.selector);
        idrx.addMinter(address(0));
    }

    function test_RemoveMinter() public {
        idrx.addMinter(minter);
        assertTrue(idrx.minters(minter));

        idrx.removeMinter(minter);

        assertFalse(idrx.minters(minter));
    }

    function test_RemoveMinterEmitsEvent() public {
        idrx.addMinter(minter);

        vm.expectEmit(true, false, false, false);
        emit MockIDRX.MinterRemoved(minter);

        idrx.removeMinter(minter);
    }

    // ============ Faucet Tests ============

    function test_Faucet() public {
        vm.prank(alice);
        idrx.faucet();

        uint256 expectedAmount = 10_000_000 * (10 ** idrx.decimals());
        assertEq(idrx.balanceOf(alice), expectedAmount);
    }

    function test_FaucetMultipleTimes() public {
        vm.startPrank(alice);
        idrx.faucet();
        idrx.faucet();
        vm.stopPrank();

        uint256 expectedAmount = 2 * 10_000_000 * (10 ** idrx.decimals());
        assertEq(idrx.balanceOf(alice), expectedAmount);
    }

    function test_FaucetTo() public {
        uint256 amount = 5000;
        idrx.faucetTo(alice, amount);

        uint256 expectedAmount = amount * (10 ** idrx.decimals());
        assertEq(idrx.balanceOf(alice), expectedAmount);
    }

    function test_RevertFaucetToNotMinter() public {
        vm.prank(alice);
        vm.expectRevert(MockIDRX.NotMinter.selector);
        idrx.faucetTo(bob, 1000);
    }

    function test_RevertFaucetToZeroAddress() public {
        vm.expectRevert(MockIDRX.InvalidAddress.selector);
        idrx.faucetTo(address(0), 1000);
    }

    // ============ Burn Tests ============

    function test_Burn() public {
        uint256 mintAmount = 1000 * 10 ** idrx.decimals();
        idrx.mint(alice, mintAmount);

        uint256 burnAmount = 500 * 10 ** idrx.decimals();
        vm.prank(alice);
        idrx.burn(burnAmount);

        assertEq(idrx.balanceOf(alice), mintAmount - burnAmount);
    }

    function test_BurnFrom() public {
        uint256 mintAmount = 1000 * 10 ** idrx.decimals();
        idrx.mint(alice, mintAmount);

        uint256 burnAmount = 500 * 10 ** idrx.decimals();
        vm.prank(alice);
        idrx.approve(owner, burnAmount);

        idrx.burnFrom(alice, burnAmount);

        assertEq(idrx.balanceOf(alice), mintAmount - burnAmount);
    }

    // ============ Transfer Tests ============

    function test_Transfer() public {
        uint256 amount = 1000 * 10 ** idrx.decimals();
        idrx.mint(alice, amount);

        uint256 transferAmount = 500 * 10 ** idrx.decimals();
        vm.prank(alice);
        idrx.transfer(bob, transferAmount);

        assertEq(idrx.balanceOf(alice), amount - transferAmount);
        assertEq(idrx.balanceOf(bob), transferAmount);
    }

    function test_TransferFrom() public {
        uint256 amount = 1000 * 10 ** idrx.decimals();
        idrx.mint(alice, amount);

        uint256 transferAmount = 500 * 10 ** idrx.decimals();
        vm.prank(alice);
        idrx.approve(bob, transferAmount);

        vm.prank(bob);
        idrx.transferFrom(alice, bob, transferAmount);

        assertEq(idrx.balanceOf(alice), amount - transferAmount);
        assertEq(idrx.balanceOf(bob), transferAmount);
    }
}
