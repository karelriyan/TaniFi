// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

<<<<<<< ours
<<<<<<< ours
/// @title TaniVault
/// @notice Minimal vault for deposits and simple project bookkeeping.
/// @dev Owner can create projects; anyone can deposit ETH.
contract TaniVault {
    address public owner;
    uint256 public totalDeposited;

    struct Project {
        uint256 amount;
        uint256 timestamp;
    }

    mapping(uint256 => Project) public projects;
    uint256 public projectCount;

    event Deposit(address indexed investor, uint256 amount);
    event ProjectCreated(uint256 indexed projectId, uint256 amount);

    modifier onlyOwner() {
        _onlyOwner();
        _;
    }

    function _onlyOwner() internal view {
        require(msg.sender == owner, "Not authorized");
    }

    constructor() {
        owner = msg.sender;
    }

    function deposit() external payable {
        require(msg.value > 0, "Invalid amount");
        totalDeposited += msg.value;
        emit Deposit(msg.sender, msg.value);
    }

    function createProject(uint256 amount) external onlyOwner {
        require(amount > 0, "Invalid amount");

        projects[projectCount] = Project({
            amount: amount,
            timestamp: block.timestamp
        });

        emit ProjectCreated(projectCount, amount);
        projectCount++;
    }

    function getBalance() external view returns (uint256) {
        return address(this).balance;
=======
=======
>>>>>>> theirs
import {IERC20} from "../lib/forge-std/src/interfaces/IERC20.sol";

/// @title TaniVault - Sharia-Compliant Agricultural Finance Protocol
/// @notice Core vault implementing Musyarakah (profit-sharing) for agricultural projects
/// @dev Manages pooling of investor funds, project lifecycle, and profit distribution
/// @author TaniFi Team - Lisk Builders Challenge 2026

contract TaniVault {
    // ============ Enums ============

    enum ProjectState {
        FUNDRAISING,    // Project is accepting investments
        ACTIVE,         // Funds disbursed, farming in progress
        HARVESTED,      // Harvest completed, awaiting finalization
        FAILED,         // Project failed (natural disaster, etc.)
        COMPLETED       // Profit distributed, project closed
    }

    // ============ Structs ============

    struct Project {
        uint256 id;
        address farmer;           // Farmer's wallet (Smart Contract Wallet)
        address cooperative;      // Cooperative/Off-taker who validates
        address approvedVendor;   // Whitelisted vendor for input purchases
        uint256 targetAmount;     // Target funding amount (in IDRX wei)
        uint256 fundedAmount;     // Currently funded amount
        uint256 farmerShareBps;   // Farmer profit share (basis points, e.g., 3000 = 30%)
        uint256 investorShareBps; // Investor profit share (e.g., 7000 = 70%)
        uint256 startTime;        // When project was funded and started
        uint256 harvestTime;      // Expected harvest date
        uint256 harvestRevenue;   // Actual harvest revenue (set during finalization)
        ProjectState state;
        string ipfsMetadata;      // IPFS hash for project documentation
    }

    // ============ State Variables ============

    // Access Control
    address public owner;
    mapping(address => bool) public cooperatives;
    mapping(address => bool) public operators;

    // Token
    IERC20 public stablecoin;  // IDRX token

    // Projects
    mapping(uint256 => Project) public projects;
    uint256 public projectCount;

    // Investments tracking: projectId => investor => amount
    mapping(uint256 => mapping(address => uint256)) public investments;

    // Investor returns: projectId => investor => claimable amount
    mapping(uint256 => mapping(address => uint256)) public investorReturns;

    // Total invested per project
    mapping(uint256 => uint256) public totalInvested;

    // Platform
    uint256 public constant PLATFORM_FEE_BPS = 100; // 1%
    uint256 public constant BPS_DENOMINATOR = 10000;
    address public treasury;

    // Security
    bool private _locked;
    bool public paused;

    // ============ Events ============

    event ProjectCreated(
        uint256 indexed projectId,
        address indexed farmer,
        address cooperative,
        uint256 targetAmount
    );

    event InvestmentReceived(
        uint256 indexed projectId,
        address indexed investor,
        uint256 amount
    );

    event FundsDisbursed(
        uint256 indexed projectId,
        address indexed vendor,
        uint256 amount
    );

    event HarvestReported(
        uint256 indexed projectId,
        uint256 revenue
    );

    event ProfitDistributed(
        uint256 indexed projectId,
        uint256 farmerShare,
        uint256 investorPool,
        uint256 platformFee
    );

    event InvestorWithdrawal(
        uint256 indexed projectId,
        address indexed investor,
        uint256 amount
    );

    event ProjectFailed(
        uint256 indexed projectId,
        string reason
    );

    event CooperativeAdded(address indexed cooperative);
    event CooperativeRemoved(address indexed cooperative);
    event OperatorAdded(address indexed operator);
    event OperatorRemoved(address indexed operator);

    // ============ Modifiers ============

    modifier onlyOwner() {
        require(msg.sender == owner, "TaniVault: not owner");
        _;
    }

    modifier onlyCooperative() {
        require(cooperatives[msg.sender], "TaniVault: not cooperative");
        _;
    }

    modifier onlyOperator() {
        require(operators[msg.sender] || msg.sender == owner, "TaniVault: not operator");
        _;
    }

    modifier nonReentrant() {
        require(!_locked, "TaniVault: reentrant call");
        _locked = true;
        _;
        _locked = false;
    }

    modifier whenNotPaused() {
        require(!paused, "TaniVault: paused");
        _;
    }

    // ============ Constructor ============

    constructor(address _stablecoin, address _treasury) {
        require(_stablecoin != address(0), "TaniVault: invalid stablecoin");
        require(_treasury != address(0), "TaniVault: invalid treasury");

        owner = msg.sender;
        stablecoin = IERC20(_stablecoin);
        treasury = _treasury;
        operators[msg.sender] = true;
    }

    // ============ Admin Functions ============

    function addCooperative(address _cooperative) external onlyOwner {
        require(_cooperative != address(0), "TaniVault: invalid address");
        cooperatives[_cooperative] = true;
        emit CooperativeAdded(_cooperative);
    }

    function removeCooperative(address _cooperative) external onlyOwner {
        cooperatives[_cooperative] = false;
        emit CooperativeRemoved(_cooperative);
    }

    function addOperator(address _operator) external onlyOwner {
        require(_operator != address(0), "TaniVault: invalid address");
        operators[_operator] = true;
        emit OperatorAdded(_operator);
    }

    function removeOperator(address _operator) external onlyOwner {
        operators[_operator] = false;
        emit OperatorRemoved(_operator);
    }

    function pause() external onlyOwner {
        paused = true;
    }

    function unpause() external onlyOwner {
        paused = false;
    }

    function setTreasury(address _treasury) external onlyOwner {
        require(_treasury != address(0), "TaniVault: invalid treasury");
        treasury = _treasury;
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "TaniVault: invalid owner");
        owner = newOwner;
    }

    // ============ Project Lifecycle Functions ============

    /// @notice Create a new agricultural project (called by cooperative)
    /// @param _farmer Address of the farmer's wallet
    /// @param _approvedVendor Whitelisted vendor for purchasing inputs
    /// @param _targetAmount Target funding amount in IDRX
    /// @param _farmerShareBps Farmer's profit share in basis points
    /// @param _harvestTime Expected harvest timestamp
    /// @param _ipfsMetadata IPFS hash containing project documentation
    function createProject(
        address _farmer,
        address _approvedVendor,
        uint256 _targetAmount,
        uint256 _farmerShareBps,
        uint256 _harvestTime,
        string calldata _ipfsMetadata
    ) external onlyCooperative whenNotPaused returns (uint256) {
        require(_farmer != address(0), "TaniVault: invalid farmer");
        require(_approvedVendor != address(0), "TaniVault: invalid vendor");
        require(_targetAmount > 0, "TaniVault: invalid target");
        require(_farmerShareBps <= 5000, "TaniVault: farmer share too high"); // Max 50%
        require(_harvestTime > block.timestamp, "TaniVault: invalid harvest time");

        uint256 projectId = projectCount;
        uint256 investorShareBps = BPS_DENOMINATOR - _farmerShareBps - PLATFORM_FEE_BPS;

        projects[projectId] = Project({
            id: projectId,
            farmer: _farmer,
            cooperative: msg.sender,
            approvedVendor: _approvedVendor,
            targetAmount: _targetAmount,
            fundedAmount: 0,
            farmerShareBps: _farmerShareBps,
            investorShareBps: investorShareBps,
            startTime: 0,
            harvestTime: _harvestTime,
            harvestRevenue: 0,
            state: ProjectState.FUNDRAISING,
            ipfsMetadata: _ipfsMetadata
        });

        projectCount++;

        emit ProjectCreated(projectId, _farmer, msg.sender, _targetAmount);
        return projectId;
    }

    /// @notice Invest in a project (public - any investor can call)
    /// @param _projectId ID of the project to invest in
    /// @param _amount Amount of IDRX to invest
    function invest(uint256 _projectId, uint256 _amount)
        external
        nonReentrant
        whenNotPaused
    {
        Project storage project = projects[_projectId];

        require(project.state == ProjectState.FUNDRAISING, "TaniVault: not fundraising");
        require(_amount > 0, "TaniVault: invalid amount");
        require(
            project.fundedAmount + _amount <= project.targetAmount,
            "TaniVault: exceeds target"
        );

        // Transfer IDRX from investor to vault
        require(
            stablecoin.transferFrom(msg.sender, address(this), _amount),
            "TaniVault: transfer failed"
        );

        // Update state
        project.fundedAmount += _amount;
        investments[_projectId][msg.sender] += _amount;
        totalInvested[_projectId] += _amount;

        emit InvestmentReceived(_projectId, msg.sender, _amount);

        // Auto-activate if fully funded
        if (project.fundedAmount == project.targetAmount) {
            _activateProject(_projectId);
        }
    }

    /// @dev Internal function to activate a fully funded project
    function _activateProject(uint256 _projectId) internal {
        Project storage project = projects[_projectId];
        project.state = ProjectState.ACTIVE;
        project.startTime = block.timestamp;
    }

    /// @notice Disburse funds to approved vendor (called by cooperative or operator)
    /// @param _projectId ID of the project
    function disburseToVendor(uint256 _projectId)
        external
        nonReentrant
        whenNotPaused
    {
        Project storage project = projects[_projectId];

        require(
            msg.sender == project.cooperative || operators[msg.sender],
            "TaniVault: not authorized"
        );
        require(project.state == ProjectState.ACTIVE, "TaniVault: not active");
        require(project.fundedAmount > 0, "TaniVault: no funds");

        uint256 disbursementAmount = project.fundedAmount;

        // Transfer to approved vendor (closed-loop - not directly to farmer)
        require(
            stablecoin.transfer(project.approvedVendor, disbursementAmount),
            "TaniVault: disbursement failed"
        );

        emit FundsDisbursed(_projectId, project.approvedVendor, disbursementAmount);
    }

    /// @notice Report harvest revenue (called by cooperative)
    /// @param _projectId ID of the project
    /// @param _revenue Total revenue from harvest sale
    function reportHarvest(uint256 _projectId, uint256 _revenue)
        external
        onlyCooperative
    {
        Project storage project = projects[_projectId];

        require(project.cooperative == msg.sender, "TaniVault: wrong cooperative");
        require(project.state == ProjectState.ACTIVE, "TaniVault: not active");

        project.harvestRevenue = _revenue;
        project.state = ProjectState.HARVESTED;

        emit HarvestReported(_projectId, _revenue);
    }

    /// @notice Finalize harvest and distribute profits (Musyarakah settlement)
    /// @param _projectId ID of the project
    function finalizeHarvest(uint256 _projectId)
        external
        nonReentrant
        whenNotPaused
    {
        Project storage project = projects[_projectId];

        require(
            msg.sender == project.cooperative || operators[msg.sender],
            "TaniVault: not authorized"
        );
        require(project.state == ProjectState.HARVESTED, "TaniVault: not harvested");
        require(project.harvestRevenue > 0, "TaniVault: no revenue");

        uint256 totalRevenue = project.harvestRevenue;

        // Calculate shares using Musyarakah profit-sharing
        uint256 platformFee = (totalRevenue * PLATFORM_FEE_BPS) / BPS_DENOMINATOR;
        uint256 farmerShare = (totalRevenue * project.farmerShareBps) / BPS_DENOMINATOR;
        uint256 investorPool = totalRevenue - farmerShare - platformFee;

        // Cooperative must deposit the revenue first
        require(
            stablecoin.transferFrom(msg.sender, address(this), totalRevenue),
            "TaniVault: revenue transfer failed"
        );

        // Transfer farmer share immediately
        require(
            stablecoin.transfer(project.farmer, farmerShare),
            "TaniVault: farmer transfer failed"
        );

        // Transfer platform fee to treasury
        require(
            stablecoin.transfer(treasury, platformFee),
            "TaniVault: fee transfer failed"
        );

        // Calculate and store investor returns proportionally
        _distributeInvestorReturns(_projectId, investorPool);

        project.state = ProjectState.COMPLETED;

        emit ProfitDistributed(_projectId, farmerShare, investorPool, platformFee);
    }

    /// @dev Internal function to distribute returns to investors based on their share
    function _distributeInvestorReturns(uint256 _projectId, uint256 _investorPool) internal {
        // Note: In production, this would iterate through investors or use a merkle tree
        // For MVP, investors can claim their proportional share
        // Their return = (their investment / total invested) * investorPool
        // This is calculated at withdrawal time
    }

    /// @notice Withdraw investor returns after project completion
    /// @param _projectId ID of the completed project
    function withdrawReturns(uint256 _projectId)
        external
        nonReentrant
    {
        Project storage project = projects[_projectId];
        require(project.state == ProjectState.COMPLETED, "TaniVault: not completed");

        uint256 invested = investments[_projectId][msg.sender];
        require(invested > 0, "TaniVault: no investment");

        // Calculate proportional return
        uint256 totalRev = project.harvestRevenue;
        uint256 platformFee = (totalRev * PLATFORM_FEE_BPS) / BPS_DENOMINATOR;
        uint256 farmerShare = (totalRev * project.farmerShareBps) / BPS_DENOMINATOR;
        uint256 investorPool = totalRev - farmerShare - platformFee;

        uint256 returnAmount = (invested * investorPool) / totalInvested[_projectId];

        // Clear investment to prevent double withdrawal
        investments[_projectId][msg.sender] = 0;

        // Transfer returns
        require(
            stablecoin.transfer(msg.sender, returnAmount),
            "TaniVault: withdrawal failed"
        );

        emit InvestorWithdrawal(_projectId, msg.sender, returnAmount);
    }

    /// @notice Mark project as failed (e.g., natural disaster)
    /// @param _projectId ID of the project
    /// @param _reason Reason for failure
    function markProjectFailed(uint256 _projectId, string calldata _reason)
        external
        onlyOperator
    {
        Project storage project = projects[_projectId];
        require(
            project.state == ProjectState.FUNDRAISING ||
            project.state == ProjectState.ACTIVE,
            "TaniVault: invalid state"
        );

        project.state = ProjectState.FAILED;

        emit ProjectFailed(_projectId, _reason);
    }

    /// @notice Refund investors if project fails during fundraising
    /// @param _projectId ID of the failed project
    function refundInvestors(uint256 _projectId)
        external
        nonReentrant
    {
        Project storage project = projects[_projectId];
        require(project.state == ProjectState.FAILED, "TaniVault: not failed");

        uint256 invested = investments[_projectId][msg.sender];
        require(invested > 0, "TaniVault: no investment");

        // Clear investment
        investments[_projectId][msg.sender] = 0;
        project.fundedAmount -= invested;

        // Refund
        require(
            stablecoin.transfer(msg.sender, invested),
            "TaniVault: refund failed"
        );
    }

    // ============ View Functions ============

    function getProject(uint256 _projectId)
        external
        view
        returns (Project memory)
    {
        return projects[_projectId];
    }

    function getInvestment(uint256 _projectId, address _investor)
        external
        view
        returns (uint256)
    {
        return investments[_projectId][_investor];
    }

    function getVaultBalance() external view returns (uint256) {
        return stablecoin.balanceOf(address(this));
    }

    function isCooperative(address _addr) external view returns (bool) {
        return cooperatives[_addr];
    }

    function isOperator(address _addr) external view returns (bool) {
        return operators[_addr];
    }

    /// @notice Calculate expected returns for an investor
    /// @param _projectId ID of the project
    /// @param _investor Address of the investor
    /// @param _expectedRevenue Expected harvest revenue
    function calculateExpectedReturns(
        uint256 _projectId,
        address _investor,
        uint256 _expectedRevenue
    ) external view returns (uint256) {
        uint256 invested = investments[_projectId][_investor];
        if (invested == 0 || totalInvested[_projectId] == 0) return 0;

        uint256 platformFee = (_expectedRevenue * PLATFORM_FEE_BPS) / BPS_DENOMINATOR;
        Project storage project = projects[_projectId];
        uint256 farmerShare = (_expectedRevenue * project.farmerShareBps) / BPS_DENOMINATOR;
        uint256 investorPool = _expectedRevenue - farmerShare - platformFee;

        return (invested * investorPool) / totalInvested[_projectId];
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
    }
}
