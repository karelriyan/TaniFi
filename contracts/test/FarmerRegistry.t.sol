// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console} from "forge-std/Test.sol";
import {FarmerRegistry} from "../src/FarmerRegistry.sol";

contract FarmerRegistryTest is Test {
    FarmerRegistry public registry;

    address public owner = address(this);
    address public farmer1 = address(0x1);
    address public farmer2 = address(0x2);
    address public kycAdmin = address(0x3);
    address public taniVault = address(0x4);

    bytes32 public phoneHash1 = keccak256("08123456789");
    bytes32 public phoneHash2 = keccak256("08234567890");
    string public metadataURI = "ipfs://QmTest123";

    function setUp() public {
        registry = new FarmerRegistry();
        registry.setTaniVault(taniVault);
    }

    // ============ Constructor Tests ============

    function test_Constructor() public view {
        assertEq(registry.name(), "TaniFi Farmer Identity");
        assertEq(registry.symbol(), "TANI-ID");
        assertEq(registry.owner(), owner);
        assertTrue(registry.kycAdmins(owner));
    }

    function test_Constants() public view {
        assertEq(registry.INITIAL_REPUTATION(), 100);
        assertEq(registry.MAX_REPUTATION(), 1000);
        assertEq(registry.PROJECT_SUCCESS_BONUS(), 10);
        assertEq(registry.PROJECT_FAILURE_PENALTY(), 20);
    }

    // ============ Registration Tests ============

    function test_RegisterFarmer() public {
        uint256 tokenId = registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        assertEq(tokenId, 1);
        assertEq(registry.ownerOf(tokenId), farmer1);
        assertEq(registry.tokenOfOwner(farmer1), tokenId);
        assertEq(registry.phoneHashToWallet(phoneHash1), farmer1);
        assertTrue(registry.isRegistered(farmer1));
    }

    function test_RegisterFarmerEmitsEvent() public {
        vm.expectEmit(true, true, false, true);
        emit FarmerRegistry.FarmerRegistered(1, farmer1, phoneHash1);

        registry.registerFarmer(farmer1, phoneHash1, metadataURI);
    }

    function test_RegisterFarmerProfile() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        FarmerRegistry.FarmerProfile memory profile = registry.getFarmerProfile(farmer1);

        assertEq(profile.tokenId, 1);
        assertEq(profile.walletAddress, farmer1);
        assertEq(profile.phoneHash, phoneHash1);
        assertEq(profile.reputationScore, 100);
        assertFalse(profile.isKYCVerified);
        assertEq(profile.completedProjects, 0);
        assertEq(profile.failedProjects, 0);
        assertEq(profile.metadataURI, metadataURI);
    }

    function test_RevertRegisterNotKYCAdmin() public {
        vm.prank(farmer1);
        vm.expectRevert(FarmerRegistry.NotKYCAdmin.selector);
        registry.registerFarmer(farmer2, phoneHash2, metadataURI);
    }

    function test_RevertRegisterZeroAddress() public {
        vm.expectRevert(FarmerRegistry.InvalidAddress.selector);
        registry.registerFarmer(address(0), phoneHash1, metadataURI);
    }

    function test_RevertRegisterAlreadyRegistered() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        vm.expectRevert(FarmerRegistry.AlreadyRegistered.selector);
        registry.registerFarmer(farmer1, phoneHash2, metadataURI);
    }

    function test_RevertRegisterPhoneAlreadyUsed() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        vm.expectRevert(FarmerRegistry.PhoneAlreadyUsed.selector);
        registry.registerFarmer(farmer2, phoneHash1, metadataURI);
    }

    // ============ KYC Verification Tests ============

    function test_VerifyFarmer() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);
        assertFalse(registry.isVerified(farmer1));

        registry.verifyFarmer(farmer1);

        assertTrue(registry.isVerified(farmer1));
    }

    function test_VerifyFarmerEmitsEvent() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        vm.expectEmit(true, false, false, true);
        emit FarmerRegistry.FarmerVerified(farmer1, 1);

        registry.verifyFarmer(farmer1);
    }

    function test_RevertVerifyNotRegistered() public {
        vm.expectRevert(FarmerRegistry.FarmerNotRegistered.selector);
        registry.verifyFarmer(farmer1);
    }

    function test_RevertVerifyAlreadyVerified() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);
        registry.verifyFarmer(farmer1);

        vm.expectRevert(FarmerRegistry.AlreadyVerified.selector);
        registry.verifyFarmer(farmer1);
    }

    function test_RevokeVerification() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);
        registry.verifyFarmer(farmer1);
        assertTrue(registry.isVerified(farmer1));

        registry.revokeVerification(farmer1);

        assertFalse(registry.isVerified(farmer1));
    }

    // ============ Soulbound Token Tests ============

    function test_RevertTransfer() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);
        uint256 tokenId = registry.tokenOfOwner(farmer1);

        vm.prank(farmer1);
        vm.expectRevert(FarmerRegistry.SoulboundTransferBlocked.selector);
        registry.transferFrom(farmer1, farmer2, tokenId);
    }

    function test_RevertSafeTransfer() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);
        uint256 tokenId = registry.tokenOfOwner(farmer1);

        vm.prank(farmer1);
        vm.expectRevert(FarmerRegistry.SoulboundTransferBlocked.selector);
        registry.safeTransferFrom(farmer1, farmer2, tokenId);
    }

    // ============ Reputation Tests ============

    function test_RecordProjectSuccess() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        vm.prank(taniVault);
        registry.recordProjectCompletion(farmer1, true);

        FarmerRegistry.FarmerProfile memory profile = registry.getFarmerProfile(farmer1);
        assertEq(profile.reputationScore, 110); // 100 + 10
        assertEq(profile.completedProjects, 1);
    }

    function test_RecordProjectFailure() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        vm.prank(taniVault);
        registry.recordProjectCompletion(farmer1, false);

        FarmerRegistry.FarmerProfile memory profile = registry.getFarmerProfile(farmer1);
        assertEq(profile.reputationScore, 80); // 100 - 20
        assertEq(profile.failedProjects, 1);
    }

    function test_ReputationCappedAtMax() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        // Complete many projects to reach max reputation
        vm.startPrank(taniVault);
        for (uint256 i = 0; i < 100; i++) {
            registry.recordProjectCompletion(farmer1, true);
        }
        vm.stopPrank();

        assertEq(registry.getReputation(farmer1), 1000); // Capped at MAX_REPUTATION
    }

    function test_ReputationFlooredAtZero() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        // Fail many projects to reach zero
        vm.startPrank(taniVault);
        for (uint256 i = 0; i < 10; i++) {
            registry.recordProjectCompletion(farmer1, false);
        }
        vm.stopPrank();

        assertEq(registry.getReputation(farmer1), 0);
    }

    function test_RevertRecordProjectNotTaniVault() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        vm.prank(farmer1);
        vm.expectRevert(FarmerRegistry.NotTaniVault.selector);
        registry.recordProjectCompletion(farmer1, true);
    }

    function test_AdjustReputation() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        registry.adjustReputation(farmer1, 500, "Manual adjustment");

        assertEq(registry.getReputation(farmer1), 500);
    }

    function test_AdjustReputationEmitsEvent() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        vm.expectEmit(true, false, false, true);
        emit FarmerRegistry.ReputationUpdated(farmer1, 100, 500, "Manual adjustment");

        registry.adjustReputation(farmer1, 500, "Manual adjustment");
    }

    function test_RevertAdjustReputationTooHigh() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        vm.expectRevert(FarmerRegistry.ScoreTooHigh.selector);
        registry.adjustReputation(farmer1, 1001, "Too high");
    }

    // ============ Metadata Tests ============

    function test_UpdateMetadata() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);
        string memory newURI = "ipfs://QmNewTest456";

        registry.updateMetadata(farmer1, newURI);

        assertEq(registry.tokenURI(1), newURI);
    }

    function test_TokenURI() public {
        registry.registerFarmer(farmer1, phoneHash1, metadataURI);

        assertEq(registry.tokenURI(1), metadataURI);
    }

    // ============ Admin Tests ============

    function test_AddKYCAdmin() public {
        assertFalse(registry.kycAdmins(kycAdmin));

        registry.addKYCAdmin(kycAdmin);

        assertTrue(registry.kycAdmins(kycAdmin));
    }

    function test_RemoveKYCAdmin() public {
        registry.addKYCAdmin(kycAdmin);
        assertTrue(registry.kycAdmins(kycAdmin));

        registry.removeKYCAdmin(kycAdmin);

        assertFalse(registry.kycAdmins(kycAdmin));
    }

    function test_SetTaniVault() public {
        address newVault = address(0x999);

        registry.setTaniVault(newVault);

        assertEq(registry.taniVault(), newVault);
    }

    function test_RevertSetTaniVaultZeroAddress() public {
        vm.expectRevert(FarmerRegistry.InvalidAddress.selector);
        registry.setTaniVault(address(0));
    }

    // ============ View Functions Tests ============

    function test_TotalSupply() public {
        assertEq(registry.totalSupply(), 0);

        registry.registerFarmer(farmer1, phoneHash1, metadataURI);
        assertEq(registry.totalSupply(), 1);

        registry.registerFarmer(farmer2, phoneHash2, metadataURI);
        assertEq(registry.totalSupply(), 2);
    }

    function test_GetReputationNotRegistered() public view {
        assertEq(registry.getReputation(farmer1), 0);
    }

    function test_IsRegisteredFalse() public view {
        assertFalse(registry.isRegistered(farmer1));
    }

    function test_IsVerifiedFalse() public view {
        assertFalse(registry.isVerified(farmer1));
    }
}
