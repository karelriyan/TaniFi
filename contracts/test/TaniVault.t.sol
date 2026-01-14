// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console} from "forge-std/Test.sol";
import {TaniVault} from "../src/TaniVault.sol";
import {MockIDRX} from "../src/MockIDRX.sol";

contract TaniVaultTest is Test {
    TaniVault public vault;
    MockIDRX public idrx;

    address public owner = address(this);
    address public cooperative = address(0x1);
    address public farmer = address(0x2);
    address public vendor = address(0x3);
    address public investor1 = address(0x4);
    address public investor2 = address(0x5);
    address public operator = address(0x6);
    address public treasury = address(0x7);

    uint256 public constant TARGET_AMOUNT = 10_000_000 * 100; // 10M IDRX (2 decimals)
    uint256 public constant FARMER_SHARE_BPS = 3000; // 30%
    uint256 public harvestTime;

    event ProjectCreated(uint256 indexed projectId, address indexed farmer, address cooperative, uint256 targetAmount);
    event InvestmentReceived(uint256 indexed projectId, address indexed investor, uint256 amount);
    event FundsDisbursed(uint256 indexed projectId, address indexed vendor, uint256 amount);
    event HarvestReported(uint256 indexed projectId, uint256 revenue);
    event ProfitDistributed(uint256 indexed projectId, uint256 farmerShare, uint256 investorPool, uint256 platformFee);
    event InvestorWithdrawal(uint256 indexed projectId, address indexed investor, uint256 amount);

    function setUp() public {
        idrx = new MockIDRX();
        vault = new TaniVault(address(idrx), treasury);

        // Setup roles
        vault.addCooperative(cooperative);
        vault.addOperator(operator);

        // Set harvest time to 90 days from now
        harvestTime = block.timestamp + 90 days;

        // Fund investors
        idrx.mint(investor1, TARGET_AMOUNT);
        idrx.mint(investor2, TARGET_AMOUNT);
    }

    // ============ Constructor Tests ============

    function test_Constructor() public view {
        assertEq(address(vault.stablecoin()), address(idrx));
        assertEq(vault.treasury(), treasury);
        assertEq(vault.owner(), owner);
        assertTrue(vault.operators(owner));
    }

    function test_RevertConstructorZeroStablecoin() public {
        vm.expectRevert(TaniVault.InvalidAddress.selector);
        new TaniVault(address(0), treasury);
    }

    function test_RevertConstructorZeroTreasury() public {
        vm.expectRevert(TaniVault.InvalidAddress.selector);
        new TaniVault(address(idrx), address(0));
    }

    function test_Constants() public view {
        assertEq(vault.PLATFORM_FEE_BPS(), 100);
        assertEq(vault.BPS_DENOMINATOR(), 10000);
    }

    // ============ Admin Functions Tests ============

    function test_AddCooperative() public {
        address newCoop = address(0x100);
        assertFalse(vault.cooperatives(newCoop));

        vault.addCooperative(newCoop);

        assertTrue(vault.cooperatives(newCoop));
    }

    function test_RemoveCooperative() public {
        vault.removeCooperative(cooperative);
        assertFalse(vault.cooperatives(cooperative));
    }

    function test_AddOperator() public {
        address newOp = address(0x101);
        vault.addOperator(newOp);
        assertTrue(vault.operators(newOp));
    }

    function test_RemoveOperator() public {
        vault.removeOperator(operator);
        assertFalse(vault.operators(operator));
    }

    function test_Pause() public {
        vault.pause();
        assertTrue(vault.paused());
    }

    function test_Unpause() public {
        vault.pause();
        vault.unpause();
        assertFalse(vault.paused());
    }

    function test_SetTreasury() public {
        address newTreasury = address(0x999);
        vault.setTreasury(newTreasury);
        assertEq(vault.treasury(), newTreasury);
    }

    // ============ Project Creation Tests ============

    function test_CreateProject() public {
        vm.prank(cooperative);
        uint256 projectId = vault.createProject(
            farmer,
            vendor,
            TARGET_AMOUNT,
            FARMER_SHARE_BPS,
            harvestTime,
            "ipfs://metadata"
        );

        assertEq(projectId, 0);
        assertEq(vault.projectCount(), 1);

        TaniVault.Project memory project = vault.getProject(projectId);
        assertEq(project.farmer, farmer);
        assertEq(project.cooperative, cooperative);
        assertEq(project.approvedVendor, vendor);
        assertEq(project.targetAmount, TARGET_AMOUNT);
        assertEq(project.fundedAmount, 0);
        assertEq(project.farmerShareBps, FARMER_SHARE_BPS);
        assertEq(project.investorShareBps, 10000 - FARMER_SHARE_BPS - 100); // 6900 = 69%
        assertEq(uint256(project.state), uint256(TaniVault.ProjectState.FUNDRAISING));
    }

    function test_CreateProjectEmitsEvent() public {
        vm.expectEmit(true, true, false, true);
        emit ProjectCreated(0, farmer, cooperative, TARGET_AMOUNT);

        vm.prank(cooperative);
        vault.createProject(farmer, vendor, TARGET_AMOUNT, FARMER_SHARE_BPS, harvestTime, "ipfs://metadata");
    }

    function test_RevertCreateProjectNotCooperative() public {
        vm.prank(investor1);
        vm.expectRevert(TaniVault.NotCooperative.selector);
        vault.createProject(farmer, vendor, TARGET_AMOUNT, FARMER_SHARE_BPS, harvestTime, "");
    }

    function test_RevertCreateProjectZeroFarmer() public {
        vm.prank(cooperative);
        vm.expectRevert(TaniVault.InvalidAddress.selector);
        vault.createProject(address(0), vendor, TARGET_AMOUNT, FARMER_SHARE_BPS, harvestTime, "");
    }

    function test_RevertCreateProjectZeroVendor() public {
        vm.prank(cooperative);
        vm.expectRevert(TaniVault.InvalidAddress.selector);
        vault.createProject(farmer, address(0), TARGET_AMOUNT, FARMER_SHARE_BPS, harvestTime, "");
    }

    function test_RevertCreateProjectZeroTarget() public {
        vm.prank(cooperative);
        vm.expectRevert(TaniVault.InvalidTarget.selector);
        vault.createProject(farmer, vendor, 0, FARMER_SHARE_BPS, harvestTime, "");
    }

    function test_RevertCreateProjectFarmerShareTooHigh() public {
        vm.prank(cooperative);
        vm.expectRevert(TaniVault.FarmerShareTooHigh.selector);
        vault.createProject(farmer, vendor, TARGET_AMOUNT, 5001, harvestTime, ""); // > 50%
    }

    function test_RevertCreateProjectInvalidHarvestTime() public {
        vm.prank(cooperative);
        vm.expectRevert(TaniVault.InvalidHarvestTime.selector);
        vault.createProject(farmer, vendor, TARGET_AMOUNT, FARMER_SHARE_BPS, block.timestamp, "");
    }

    function test_RevertCreateProjectWhenPaused() public {
        vault.pause();

        vm.prank(cooperative);
        vm.expectRevert();
        vault.createProject(farmer, vendor, TARGET_AMOUNT, FARMER_SHARE_BPS, harvestTime, "");
    }

    // ============ Investment Tests ============

    function _createProject() internal returns (uint256) {
        vm.prank(cooperative);
        return vault.createProject(farmer, vendor, TARGET_AMOUNT, FARMER_SHARE_BPS, harvestTime, "");
    }

    function test_Invest() public {
        uint256 projectId = _createProject();
        uint256 investAmount = 1_000_000 * 100; // 1M IDRX

        vm.startPrank(investor1);
        idrx.approve(address(vault), investAmount);
        vault.invest(projectId, investAmount);
        vm.stopPrank();

        assertEq(vault.getInvestment(projectId, investor1), investAmount);
        assertEq(vault.totalInvested(projectId), investAmount);
        assertEq(idrx.balanceOf(address(vault)), investAmount);

        TaniVault.Project memory project = vault.getProject(projectId);
        assertEq(project.fundedAmount, investAmount);
    }

    function test_InvestEmitsEvent() public {
        uint256 projectId = _createProject();
        uint256 investAmount = 1_000_000 * 100;

        vm.startPrank(investor1);
        idrx.approve(address(vault), investAmount);

        vm.expectEmit(true, true, false, true);
        emit InvestmentReceived(projectId, investor1, investAmount);

        vault.invest(projectId, investAmount);
        vm.stopPrank();
    }

    function test_InvestMultipleInvestors() public {
        uint256 projectId = _createProject();
        uint256 investAmount1 = 3_000_000 * 100;
        uint256 investAmount2 = 2_000_000 * 100;

        vm.startPrank(investor1);
        idrx.approve(address(vault), investAmount1);
        vault.invest(projectId, investAmount1);
        vm.stopPrank();

        vm.startPrank(investor2);
        idrx.approve(address(vault), investAmount2);
        vault.invest(projectId, investAmount2);
        vm.stopPrank();

        assertEq(vault.totalInvested(projectId), investAmount1 + investAmount2);
    }

    function test_InvestFullyFundsProject() public {
        uint256 projectId = _createProject();

        vm.startPrank(investor1);
        idrx.approve(address(vault), TARGET_AMOUNT);
        vault.invest(projectId, TARGET_AMOUNT);
        vm.stopPrank();

        TaniVault.Project memory project = vault.getProject(projectId);
        assertEq(uint256(project.state), uint256(TaniVault.ProjectState.ACTIVE));
        assertGt(project.startTime, 0);
    }

    function test_RevertInvestZeroAmount() public {
        uint256 projectId = _createProject();

        vm.prank(investor1);
        vm.expectRevert(TaniVault.InvalidAmount.selector);
        vault.invest(projectId, 0);
    }

    function test_RevertInvestExceedsTarget() public {
        uint256 projectId = _createProject();

        vm.startPrank(investor1);
        idrx.approve(address(vault), TARGET_AMOUNT + 1);
        vm.expectRevert(TaniVault.ExceedsTarget.selector);
        vault.invest(projectId, TARGET_AMOUNT + 1);
        vm.stopPrank();
    }

    function test_RevertInvestNotFundraising() public {
        uint256 projectId = _createProject();

        // Fully fund the project
        vm.startPrank(investor1);
        idrx.approve(address(vault), TARGET_AMOUNT);
        vault.invest(projectId, TARGET_AMOUNT);
        vm.stopPrank();

        // Try to invest more
        vm.startPrank(investor2);
        idrx.approve(address(vault), 100);
        vm.expectRevert(TaniVault.NotFundraising.selector);
        vault.invest(projectId, 100);
        vm.stopPrank();
    }

    // ============ Disbursement Tests ============

    function _fundProject(uint256 projectId) internal {
        vm.startPrank(investor1);
        idrx.approve(address(vault), TARGET_AMOUNT);
        vault.invest(projectId, TARGET_AMOUNT);
        vm.stopPrank();
    }

    function test_DisburseToVendor() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);

        uint256 vendorBalanceBefore = idrx.balanceOf(vendor);

        vm.prank(cooperative);
        vault.disburseToVendor(projectId);

        assertEq(idrx.balanceOf(vendor), vendorBalanceBefore + TARGET_AMOUNT);
    }

    function test_DisburseToVendorEmitsEvent() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);

        vm.expectEmit(true, true, false, true);
        emit FundsDisbursed(projectId, vendor, TARGET_AMOUNT);

        vm.prank(cooperative);
        vault.disburseToVendor(projectId);
    }

    function test_DisburseByOperator() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);

        vm.prank(operator);
        vault.disburseToVendor(projectId);

        assertEq(idrx.balanceOf(vendor), TARGET_AMOUNT);
    }

    function test_RevertDisburseNotAuthorized() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);

        vm.prank(investor1);
        vm.expectRevert(TaniVault.NotAuthorized.selector);
        vault.disburseToVendor(projectId);
    }

    function test_RevertDisburseNotActive() public {
        uint256 projectId = _createProject();
        // Project is still in FUNDRAISING state

        vm.prank(cooperative);
        vm.expectRevert(TaniVault.NotActive.selector);
        vault.disburseToVendor(projectId);
    }

    // ============ Harvest Reporting Tests ============

    function _disburseToVendor(uint256 projectId) internal {
        vm.prank(cooperative);
        vault.disburseToVendor(projectId);
    }

    function test_ReportHarvest() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);
        _disburseToVendor(projectId);

        uint256 revenue = 15_000_000 * 100; // 15M IDRX

        vm.prank(cooperative);
        vault.reportHarvest(projectId, revenue);

        TaniVault.Project memory project = vault.getProject(projectId);
        assertEq(project.harvestRevenue, revenue);
        assertEq(uint256(project.state), uint256(TaniVault.ProjectState.HARVESTED));
    }

    function test_ReportHarvestEmitsEvent() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);
        _disburseToVendor(projectId);

        uint256 revenue = 15_000_000 * 100;

        vm.expectEmit(true, false, false, true);
        emit HarvestReported(projectId, revenue);

        vm.prank(cooperative);
        vault.reportHarvest(projectId, revenue);
    }

    function test_RevertReportHarvestWrongCooperative() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);
        _disburseToVendor(projectId);

        address otherCoop = address(0x999);
        vault.addCooperative(otherCoop);

        vm.prank(otherCoop);
        vm.expectRevert(TaniVault.WrongCooperative.selector);
        vault.reportHarvest(projectId, 100);
    }

    function test_RevertReportHarvestNotActive() public {
        uint256 projectId = _createProject();

        vm.prank(cooperative);
        vm.expectRevert(TaniVault.NotActive.selector);
        vault.reportHarvest(projectId, 100);
    }

    // ============ Harvest Finalization Tests ============

    function _reportHarvest(uint256 projectId, uint256 revenue) internal {
        vm.prank(cooperative);
        vault.reportHarvest(projectId, revenue);
    }

    function test_FinalizeHarvest() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);
        _disburseToVendor(projectId);

        uint256 revenue = 15_000_000 * 100; // 15M IDRX
        _reportHarvest(projectId, revenue);

        // Cooperative needs tokens to finalize
        idrx.mint(cooperative, revenue);
        vm.prank(cooperative);
        idrx.approve(address(vault), revenue);

        uint256 farmerBalanceBefore = idrx.balanceOf(farmer);
        uint256 treasuryBalanceBefore = idrx.balanceOf(treasury);

        vm.prank(cooperative);
        vault.finalizeHarvest(projectId);

        // Calculate expected shares
        uint256 platformFee = (revenue * 100) / 10000; // 1%
        uint256 farmerShare = (revenue * 3000) / 10000; // 30%

        assertEq(idrx.balanceOf(farmer), farmerBalanceBefore + farmerShare);
        assertEq(idrx.balanceOf(treasury), treasuryBalanceBefore + platformFee);

        TaniVault.Project memory project = vault.getProject(projectId);
        assertEq(uint256(project.state), uint256(TaniVault.ProjectState.COMPLETED));
    }

    function test_FinalizeHarvestEmitsEvent() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);
        _disburseToVendor(projectId);

        uint256 revenue = 15_000_000 * 100;
        _reportHarvest(projectId, revenue);

        idrx.mint(cooperative, revenue);
        vm.prank(cooperative);
        idrx.approve(address(vault), revenue);

        uint256 platformFee = (revenue * 100) / 10000;
        uint256 farmerShare = (revenue * 3000) / 10000;
        uint256 investorPool = revenue - farmerShare - platformFee;

        vm.expectEmit(true, false, false, true);
        emit ProfitDistributed(projectId, farmerShare, investorPool, platformFee);

        vm.prank(cooperative);
        vault.finalizeHarvest(projectId);
    }

    function test_RevertFinalizeNotHarvested() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);

        vm.prank(cooperative);
        vm.expectRevert(TaniVault.NotHarvested.selector);
        vault.finalizeHarvest(projectId);
    }

    // ============ Investor Withdrawal Tests ============

    function _finalizeHarvest(uint256 projectId, uint256 revenue) internal {
        _reportHarvest(projectId, revenue);

        idrx.mint(cooperative, revenue);
        vm.prank(cooperative);
        idrx.approve(address(vault), revenue);

        vm.prank(cooperative);
        vault.finalizeHarvest(projectId);
    }

    function test_WithdrawReturns() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);
        _disburseToVendor(projectId);

        uint256 revenue = 15_000_000 * 100;
        _finalizeHarvest(projectId, revenue);

        uint256 investorBalanceBefore = idrx.balanceOf(investor1);

        vm.prank(investor1);
        vault.withdrawReturns(projectId);

        // Calculate expected return
        uint256 platformFee = (revenue * 100) / 10000;
        uint256 farmerShare = (revenue * 3000) / 10000;
        uint256 investorPool = revenue - farmerShare - platformFee;
        uint256 expectedReturn = (TARGET_AMOUNT * investorPool) / TARGET_AMOUNT; // 100% share since only investor

        assertEq(idrx.balanceOf(investor1), investorBalanceBefore + expectedReturn);
        assertEq(vault.getInvestment(projectId, investor1), 0); // Investment cleared
    }

    function test_WithdrawReturnsEmitsEvent() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);
        _disburseToVendor(projectId);

        uint256 revenue = 15_000_000 * 100;
        _finalizeHarvest(projectId, revenue);

        uint256 platformFee = (revenue * 100) / 10000;
        uint256 farmerShare = (revenue * 3000) / 10000;
        uint256 investorPool = revenue - farmerShare - platformFee;

        vm.expectEmit(true, true, false, true);
        emit InvestorWithdrawal(projectId, investor1, investorPool);

        vm.prank(investor1);
        vault.withdrawReturns(projectId);
    }

    function test_WithdrawReturnsMultipleInvestors() public {
        uint256 projectId = _createProject();

        // Two investors split investment
        uint256 invest1 = 6_000_000 * 100; // 60%
        uint256 invest2 = 4_000_000 * 100; // 40%

        vm.startPrank(investor1);
        idrx.approve(address(vault), invest1);
        vault.invest(projectId, invest1);
        vm.stopPrank();

        vm.startPrank(investor2);
        idrx.approve(address(vault), invest2);
        vault.invest(projectId, invest2);
        vm.stopPrank();

        _disburseToVendor(projectId);

        uint256 revenue = 15_000_000 * 100;
        _finalizeHarvest(projectId, revenue);

        uint256 platformFee = (revenue * 100) / 10000;
        uint256 farmerShare = (revenue * 3000) / 10000;
        uint256 investorPool = revenue - farmerShare - platformFee;

        uint256 expectedReturn1 = (invest1 * investorPool) / TARGET_AMOUNT;
        uint256 expectedReturn2 = (invest2 * investorPool) / TARGET_AMOUNT;

        uint256 balance1Before = idrx.balanceOf(investor1);
        uint256 balance2Before = idrx.balanceOf(investor2);

        vm.prank(investor1);
        vault.withdrawReturns(projectId);

        vm.prank(investor2);
        vault.withdrawReturns(projectId);

        assertEq(idrx.balanceOf(investor1), balance1Before + expectedReturn1);
        assertEq(idrx.balanceOf(investor2), balance2Before + expectedReturn2);
    }

    function test_RevertWithdrawNotCompleted() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);

        vm.prank(investor1);
        vm.expectRevert(TaniVault.NotCompleted.selector);
        vault.withdrawReturns(projectId);
    }

    function test_RevertWithdrawNoInvestment() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);
        _disburseToVendor(projectId);

        uint256 revenue = 15_000_000 * 100;
        _finalizeHarvest(projectId, revenue);

        vm.prank(investor2); // investor2 didn't invest
        vm.expectRevert(TaniVault.NoInvestment.selector);
        vault.withdrawReturns(projectId);
    }

    function test_RevertWithdrawTwice() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);
        _disburseToVendor(projectId);

        uint256 revenue = 15_000_000 * 100;
        _finalizeHarvest(projectId, revenue);

        vm.prank(investor1);
        vault.withdrawReturns(projectId);

        vm.prank(investor1);
        vm.expectRevert(TaniVault.NoInvestment.selector);
        vault.withdrawReturns(projectId);
    }

    // ============ Project Failure Tests ============

    function test_MarkProjectFailed() public {
        uint256 projectId = _createProject();

        vm.prank(operator);
        vault.markProjectFailed(projectId, "Natural disaster");

        TaniVault.Project memory project = vault.getProject(projectId);
        assertEq(uint256(project.state), uint256(TaniVault.ProjectState.FAILED));
    }

    function test_RevertMarkFailedNotOperator() public {
        uint256 projectId = _createProject();

        vm.prank(investor1);
        vm.expectRevert(TaniVault.NotOperator.selector);
        vault.markProjectFailed(projectId, "reason");
    }

    function test_RefundInvestors() public {
        uint256 projectId = _createProject();

        vm.startPrank(investor1);
        idrx.approve(address(vault), TARGET_AMOUNT / 2);
        vault.invest(projectId, TARGET_AMOUNT / 2);
        vm.stopPrank();

        uint256 balanceBefore = idrx.balanceOf(investor1);

        vm.prank(operator);
        vault.markProjectFailed(projectId, "Failed");

        vm.prank(investor1);
        vault.refundInvestors(projectId);

        assertEq(idrx.balanceOf(investor1), balanceBefore + TARGET_AMOUNT / 2);
    }

    function test_RevertRefundNotFailed() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);

        vm.prank(investor1);
        vm.expectRevert(TaniVault.NotFailed.selector);
        vault.refundInvestors(projectId);
    }

    // ============ View Functions Tests ============

    function test_GetProject() public {
        uint256 projectId = _createProject();
        TaniVault.Project memory project = vault.getProject(projectId);

        assertEq(project.id, projectId);
        assertEq(project.farmer, farmer);
    }

    function test_GetInvestment() public {
        uint256 projectId = _createProject();

        vm.startPrank(investor1);
        idrx.approve(address(vault), 1000);
        vault.invest(projectId, 1000);
        vm.stopPrank();

        assertEq(vault.getInvestment(projectId, investor1), 1000);
    }

    function test_GetVaultBalance() public {
        uint256 projectId = _createProject();

        vm.startPrank(investor1);
        idrx.approve(address(vault), 1000);
        vault.invest(projectId, 1000);
        vm.stopPrank();

        assertEq(vault.getVaultBalance(), 1000);
    }

    function test_CalculateExpectedReturns() public {
        uint256 projectId = _createProject();
        _fundProject(projectId);

        uint256 expectedRevenue = 15_000_000 * 100;
        uint256 expectedReturns = vault.calculateExpectedReturns(projectId, investor1, expectedRevenue);

        uint256 platformFee = (expectedRevenue * 100) / 10000;
        uint256 farmerShare = (expectedRevenue * 3000) / 10000;
        uint256 investorPool = expectedRevenue - farmerShare - platformFee;

        assertEq(expectedReturns, investorPool);
    }

    function test_IsCooperative() public view {
        assertTrue(vault.isCooperative(cooperative));
        assertFalse(vault.isCooperative(investor1));
    }

    function test_IsOperator() public view {
        assertTrue(vault.isOperator(operator));
        assertTrue(vault.isOperator(owner)); // Owner is also operator
        assertFalse(vault.isOperator(investor1));
    }
}
